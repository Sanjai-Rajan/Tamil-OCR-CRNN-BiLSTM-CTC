"""
Train CRNN v2 from scratch.

  .venv\\Scripts\\python -m src.train --config configs\\baseline_hw.yaml
  .venv\\Scripts\\python -m src.train --config configs\\baseline_hw.yaml --resume        (continue last.pth)

Outputs go to runs/<experiment>/:
  last.pth, best.pth, history.csv/json (real learning rate per epoch), config.yaml,
  preds_eXXX.txt (sample predictions per validation set), train.log
"""
import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import charset as cs  # noqa: E402
from src.augment import RealAugment  # noqa: E402
from src.data.datasets import (ManifestDataset, MixtureDataset, PairedSourceBatchSampler,  # noqa: E402
                               SynthSource, WidthBucketSampler, collate)
from src.metrics import Scoreboard  # noqa: E402
from src.model.crnn_v2 import CRNNv2, count_parameters  # noqa: E402


# ----------------------------------------------------------------------------- setup
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_source(spec, seed):
    t = spec["type"]
    if t == "manifest":
        aug = RealAugment(spec["augment"]) if spec.get("augment") else None
        return ManifestDataset(ROOT / spec["path"], spec["name"], augment=aug, where=spec.get("where"))
    if t == "synth":
        from src.synth.generator import SynthWordGenerator
        vocab = spec.get("vocab")
        gen = SynthWordGenerator([str(ROOT / f) for f in spec.get("fonts", ["assets/fonts"])],
                                 vocab_json=str(ROOT / vocab) if vocab else None,
                                 old_print_prob=spec.get("old_print_prob", 0.3), text_seed=seed)
        return SynthSource(gen, spec.get("length", 1_000_000), spec["name"], seed_base=seed * 7919)
    raise ValueError(f"unknown source type {t}")


def build_val(spec):
    return ManifestDataset(ROOT / spec["path"], spec["name"], augment=None,
                           max_items=spec.get("max_items"), where=spec.get("where"))


def _worker_init(_):
    # Each worker is its own process; by default OpenCV would start one thread per CPU core in EVERY worker
    # (8 workers x all cores), starving the main process that feeds the GPU. Results are identical.
    import cv2
    cv2.setNumThreads(1)


def make_loader(ds, bs, workers, shuffle=False, sampler=None):
    kw = dict(num_workers=workers, collate_fn=collate, pin_memory=torch.cuda.is_available())
    if sampler is not None:
        kw["batch_sampler"] = sampler
    else:
        kw.update(batch_size=bs, shuffle=shuffle, drop_last=False)
    if workers > 0:
        kw["prefetch_factor"] = 2
        kw["worker_init_fn"] = _worker_init
    return DataLoader(ds, **kw)


# ----------------------------------------------------------------------------- model / optimiser / one step
# Shared by this trainer and tools/bench_train.py, so the benchmark measures exactly the trainer's work.
def build_model(mc, device, channels_last=False):
    model = CRNNv2(cs.NUM_CLASSES, in_h=mc.get("in_h", 48), channels=tuple(mc.get("channels", (64, 128, 256, 384))),
                   lstm_hidden=mc.get("lstm_hidden", 256), lstm_layers=mc.get("lstm_layers", 2),
                   dropout=mc.get("dropout", 0.25)).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def make_sampler(ds, tc, train=True, seed=42):
    """Width bucketing (config: bucket_by_width, bucket_pool, max_batch_columns); None = plain random batches.

    batch_mix: "paired" builds every training batch from all sources in their configured proportions
    (Experiment 002-A). It applies to the training mixture only; validation sets keep width bucketing.
    """
    if not tc.get("bucket_by_width", False):
        return None
    if train and tc.get("batch_mix", "none") == "paired" and hasattr(ds, "locate"):
        return PairedSourceBatchSampler(ds, int(tc["batch_size"]), weights=getattr(ds, "weights", None),
                                        max_columns=tc.get("max_batch_columns"),
                                        unknown_width=tc.get("unknown_width_estimate"),
                                        shuffle=True, seed=seed)
    return WidthBucketSampler(ds, int(tc["batch_size"]), pool_batches=int(tc.get("bucket_pool", 50)),
                              max_columns=tc.get("max_batch_columns"), shuffle=train, seed=seed,
                              unknown_width_estimate=tc.get("unknown_width_estimate"))


def build_optim(model, tc, total_steps, device):
    fused = bool(tc.get("fused_adamw", False)) and device.type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc.get("weight_decay", 0.01)),
                            fused=fused)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(tc["lr"]), total_steps=total_steps,
                                                pct_start=float(tc.get("warmup_pct", 0.05)), anneal_strategy="cos",
                                                div_factor=25.0, final_div_factor=1000.0)
    return opt, sched


# GradScaler skips optimizer.step() on the first step(s) while it calibrates the loss scale; the LR scheduler
# is still stepped (as in PyTorch's own AMP recipe), which triggers this one-time, harmless warning.
warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`")


class TrainStep:
    """One optimisation step with no host<->GPU synchronisation on the CUDA path:
      * targets are moved from pinned memory asynchronously (CTCLoss would otherwise do a blocking copy)
      * no Python check of the loss value (GradScaler already skips steps whose gradients are inf/NaN)
      * no loss.item(); the loss comes back as a GPU tensor and is read only when logging
      * with fused_adamw, GradScaler hands its inf-check to the optimiser kernel instead of calling .item()
    """

    def __init__(self, model, opt, sched, scaler, device, amp, grad_clip, channels_last=False):
        self.model, self.opt, self.sched, self.scaler = model, opt, sched, scaler
        self.device, self.amp, self.grad_clip, self.cl = device, amp, grad_clip, channels_last
        self.ctc = nn.CTCLoss(blank=cs.BLANK, reduction="mean", zero_infinity=True)
        self.params = [p for p in model.parameters() if p.requires_grad]

    def __call__(self, batch):
        dev = self.device
        x = batch["images"].to(dev, non_blocking=True)
        if self.cl:
            x = x.contiguous(memory_format=torch.channels_last)
        targets = batch["targets"].to(dev, non_blocking=True)
        with torch.autocast(dev.type, enabled=self.amp, dtype=torch.float16):
            logits, lengths = self.model(x, batch["widths"])
        loss = self.ctc(logits.float().log_softmax(-1), targets, lengths, batch["target_lengths"])
        if dev.type != "cuda" and not torch.isfinite(loss):     # CPU/no-AMP only: guard against NaN updates
            self.opt.zero_grad(set_to_none=True)
            return loss.detach()
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        return loss.detach()


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, loader, device, amp, n_examples=40):
    model.eval()
    board, examples, seen = Scoreboard(), [], 0
    stride = max(1, len(loader.dataset) // max(1, n_examples))
    for batch in loader:
        if batch is None:
            continue
        x = batch["images"].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda", dtype=torch.float16):
            logits, lengths = model(x, batch["widths"])
        best = logits.float().argmax(-1).cpu()          # T x B
        for i, ref in enumerate(batch["texts"]):
            hyp = cs.decode(cs.ctc_greedy_collapse(best[: lengths[i], i].tolist()))
            board.add(ref, hyp, batch["sources"][i])
            if seen % stride == 0 and len(examples) < n_examples:
                examples.append((ref, hyp))
            seen += 1
    model.train()
    return board.summary(), examples


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from runs/<experiment>/last.pth")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after N steps")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    tc, mc = cfg["train"], cfg.get("model", {})
    seed_everything(cfg.get("seed", 42))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(tc.get("amp", True)) and device.type == "cuda"
    if device.type == "cuda":
        # Off by default: inputs have variable width, and with benchmark=True cuDNN re-tunes every conv
        # for every new width it sees (hundreds of widths -> seconds per batch for most of epoch 0).
        torch.backends.cudnn.benchmark = bool(tc.get("cudnn_benchmark", False))

    run = ROOT / "runs" / cfg["experiment"]
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run / "config.yaml")
    log_f = open(run / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    # data
    sources = [build_source(s, cfg.get("seed", 42) + i) for i, s in enumerate(cfg["sources"])]
    weights = [float(s.get("weight", 1.0)) for s in cfg["sources"]]
    for s, spec in zip(sources, cfg["sources"]):
        extra = f" (dropped {s.dropped} labels with unsupported characters)" if hasattr(s, "dropped") and s.dropped else ""
        log(f"source {spec['name']:<16} type={spec['type']:<8} weight={spec.get('weight', 1.0)} size={len(s)}{extra}")
    train_ds = MixtureDataset(sources, weights, int(tc["epoch_size"]), seed=cfg.get("seed", 42))
    vals = [(v["name"], build_val(v), v.get("primary", True)) for v in cfg.get("val", [])]
    for name, v, _ in vals:
        log(f"val    {name:<16} size={len(v)}")

    # model
    channels_last = bool(tc.get("channels_last", False)) and device.type == "cuda"
    model = build_model(mc, device, channels_last)
    log(f"model CRNNv2  params={count_parameters(model):,}  classes={cs.NUM_CLASSES}  device={device}  amp={amp}  "
        f"fused_adamw={bool(tc.get('fused_adamw', False))}  channels_last={channels_last}  "
        f"cudnn_benchmark={torch.backends.cudnn.benchmark}")

    bs, workers, epochs = int(tc["batch_size"]), int(tc.get("num_workers", 4)), int(tc["epochs"])
    train_sampler = make_sampler(train_ds, tc, train=True, seed=cfg.get("seed", 42))
    if train_sampler is None:
        steps_per_epoch = math.ceil(len(train_ds) / bs)
        total_steps = epochs * steps_per_epoch
    else:
        t = time.time()
        per_epoch = [len(train_sampler.batches(e)) for e in range(epochs)]   # exact, so OneCycleLR ends on time
        steps_per_epoch, total_steps = per_epoch[0], sum(per_epoch)
        log(f"width bucketing: {steps_per_epoch} batches in epoch 0 (plain: {math.ceil(len(train_ds) / bs)}), "
            f"max_batch_columns={tc.get('max_batch_columns')}, planned in {time.time() - t:.0f} s")
    val_samplers = {name: make_sampler(v, tc, train=False) for name, v, _ in vals}
    opt, sched = build_optim(model, tc, total_steps, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best_score, history = 0, float("inf"), []
    if args.resume and (run / "last.pth").exists():
        ck = torch.load(run / "last.pth", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_score, history = ck["epoch"] + 1, ck["best_score"], ck.get("history", [])
        log(f"resumed from epoch {ck['epoch']} (best score {best_score:.4f})")

    def save(path, epoch, metrics):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_score": best_score, "metrics": metrics,
                    "history": history, "config": cfg, "charset": cs.CHARSET,
                    "model_args": {"num_classes": cs.NUM_CLASSES, "in_h": mc.get("in_h", 48),
                                   "channels": list(mc.get("channels", (64, 128, 256, 384))),
                                   "lstm_hidden": mc.get("lstm_hidden", 256), "lstm_layers": mc.get("lstm_layers", 2),
                                   "dropout": mc.get("dropout", 0.25)}}, path)

    t_start, max_hours, step_global = time.time(), float(tc.get("max_hours", 1e9)), 0
    grad_clip = float(tc.get("grad_clip", 5.0))
    train_step = TrainStep(model, opt, sched, scaler, device, amp, grad_clip, channels_last)
    stop = False
    for epoch in range(start_epoch, epochs):
        train_ds.set_epoch(epoch)
        for s in sources:
            if hasattr(s, "set_epoch"):
                s.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            steps_per_epoch = len(train_sampler)
            if hasattr(train_sampler, "stats"):
                log(f"batch plan: {train_sampler.stats(epoch)}")
        loader = make_loader(train_ds, bs, workers, sampler=train_sampler)
        model.train()
        t0, seen, n_loss = time.time(), 0, 0
        loss_sum = torch.zeros((), device=device)          # accumulated on the GPU; read once per log line
        nonfinite = torch.zeros((), device=device, dtype=torch.long)
        comp = {"batches": 0, "single_source": 0, "images": 0, "per_source": {}}   # realised composition
        # throughput is measured over the window since the last log line, excluding the first batch
        # (worker start-up + CUDA/cuDNN initialisation), and split into data-wait vs GPU-compute time
        t_win, n_win, wait_win, t_fetch = None, 0, 0.0, time.time()
        for step, batch in enumerate(loader):
            wait = time.time() - t_fetch
            if step == 0:
                log(f"first batch ready after {wait:.1f} s (DataLoader worker start-up)")
            if batch is None:
                t_fetch = time.time()
                continue
            names = batch["sources"]                       # realised batch composition (measured, not planned)
            comp["batches"] += 1; comp["images"] += len(names)
            comp["single_source"] += int(len(set(names)) == 1)
            for nm in names:
                comp["per_source"][nm] = comp["per_source"].get(nm, 0) + 1
            loss = train_step(batch)
            finite = torch.isfinite(loss)
            loss_sum += torch.where(finite, loss, torch.zeros_like(loss)); nonfinite += (~finite).long()
            nb = batch["images"].shape[0]
            n_loss += 1; seen += nb; step_global += 1
            if step == 0:
                loss.item()                                # wait for the GPU so the timing below is real
                log(f"first step done in {time.time() - t0:.1f} s total (includes CUDA/cuDNN initialisation)")
                t_win, n_win, wait_win = time.time(), 0, 0.0
            else:
                n_win += nb; wait_win += wait
            if step and step % int(tc.get("log_every", 100)) == 0:
                cur = loss.item()                          # the only sync in the loop: once per log line
                el = max(1e-6, time.time() - t_win)
                mem = ""
                if device.type == "cuda":
                    st = torch.cuda.memory_stats()
                    mem = (f"  GPU mem peak {torch.cuda.max_memory_reserved() / 2**30:.1f} GB, "
                           f"alloc retries {st.get('num_alloc_retries', 0)}")
                log(f"ep {epoch} step {step}/{steps_per_epoch} loss {cur:.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{n_win / el:.0f} img/s  (waiting for data {100 * wait_win / el:.0f}% of the time){mem}")
                t_win, n_win, wait_win = time.time(), 0, 0.0
            if args.max_steps and step_global >= args.max_steps:
                stop = True; break
            if (time.time() - t_start) / 3600 > max_hours:
                log("time budget reached - stopping after saving"); stop = True; break
            t_fetch = time.time()

        # ---- validation
        n_bad = int(nonfinite.item())
        nb = max(1, comp["batches"]); ni = max(1, comp["images"])
        comp_summary = {"single_source_batches_pct": 100.0 * comp["single_source"] / nb,
                        "mean_batch_size": comp["images"] / nb,
                        **{f"share_{k}_pct": 100.0 * v / ni for k, v in sorted(comp["per_source"].items())}}
        log("realised composition: " + ", ".join(f"{k} {v:.1f}" for k, v in comp_summary.items()))
        metrics = {"epoch": epoch, **{f"comp_{k}": v for k, v in comp_summary.items()}, "train_loss": float(loss_sum.item()) / max(1, n_loss - n_bad),
                   "lr_end": sched.get_last_lr()[0], "nonfinite_losses": n_bad, "amp_scale": float(scaler.get_scale()),
                   "epoch_minutes": (time.time() - t0) / 60}
        prim = []
        with open(run / f"preds_e{epoch:03d}.txt", "w", encoding="utf-8") as pf:
            for name, vds, primary in vals:
                summ, ex = evaluate(model, make_loader(vds, bs, workers, sampler=val_samplers[name]), device, amp)
                a = summ["all"]
                metrics[f"{name}_cer"], metrics[f"{name}_wer"], metrics[f"{name}_lenratio"] = a["cer"], a["wer"], a["len_ratio"]
                for k, v in summ.items():
                    if k.startswith("len:"):
                        metrics[f"{name}_{k.replace(':', '')}_cer"] = v["cer"]
                if primary:
                    prim.append(a["cer"])
                pf.write(f"===== {name}  CER {a['cer']:.4f}  WER {a['wer']:.4f}\n")
                for ref, hyp in ex:
                    pf.write(f"{'OK ' if ref == hyp else 'ERR'}  GT: {ref}\t PRED: {hyp}\n")
                log(f"val {name:<16} CER {a['cer']*100:6.2f}%  WER {a['wer']*100:6.2f}%  len-ratio {a['len_ratio']:.2f}")
        score = float(np.mean(prim)) if prim else metrics["train_loss"]
        metrics["score"] = score
        history.append(metrics)
        if score < best_score:
            best_score = score
            save(run / "best.pth", epoch, metrics)
            log(f"new best (mean primary CER {score*100:.2f}%) -> best.pth")
        save(run / "last.pth", epoch, metrics)
        with open(run / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=1)
        keys = sorted({k for h in history for k in h})
        with open(run / "history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(history)
        log(f"epoch {epoch} done in {metrics['epoch_minutes']:.1f} min  train_loss {metrics['train_loss']:.4f}")
        if stop:
            break
    log("training finished")


if __name__ == "__main__":   # required on Windows (DataLoader workers use spawn)
    main()
