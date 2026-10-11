"""
Measure training speed BEFORE a long run. Local only; writes nothing (no runs/, no checkpoints, no data changes).

  .venv\\Scripts\\python tools\\bench_train.py --config configs\\baseline_hw.yaml
  .venv\\Scripts\\python tools\\bench_train.py --config configs\\baseline_hw.yaml --gpu-tests      (also G0-G4)

Default run (~10 min): the full training loop with random batches vs width-bucketed batches, a batch-32 memory
probe, validation speed, and an epoch-time estimate. Each loop test reports GPU memory, allocator retries and
how much of the batch tensor is padding.

Every GPU test uses the trainer's own code (src.train: build_model / build_optim / TrainStep / make_loader),
so what is measured is exactly what training will do. While each test runs, nvidia-smi is sampled
(utilisation, SM clock, power, temperature, throttle reasons) to tell "GPU idle" apart from "GPU slowed down".

Tests
  G0  GPU only, cached batches, bare step (same as the previous stage 3)    <- reference
  G1  GPU only, cached batches, the OLD trainer step (5 host syncs/step)     <- what the old trainer cost
  G2  GPU only, cached batches, new TrainStep (no syncs), standard AdamW
  G3  G2 + fused AdamW
  G4  G3 + channels_last
  L*  full loop (DataLoader + TrainStep): random vs width-bucketed batches, and a batch-32 memory probe
  G0' G0 again at the end                                                     <- thermal / power drift
  V   validation pass speed (subset, extrapolated to the full validation set)
"""
import argparse
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import charset as cs  # noqa: E402
from src.data.datasets import ManifestDataset, MixtureDataset  # noqa: E402
from src.data.datasets import WidthBucketSampler  # noqa: E402
from src.train import (TrainStep, build_model, build_optim, build_source, evaluate,  # noqa: E402
                       make_loader)

THROTTLE_BITS = {0x4: "power-cap", 0x8: "hw-slowdown", 0x20: "sw-thermal", 0x40: "hw-thermal", 0x80: "power-brake"}


# ----------------------------------------------------------------------------- nvidia-smi sampler
class GpuMonitor:
    FIELDS_FULL = "utilization.gpu,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active"
    FIELDS_OLD = "utilization.gpu,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active"
    FIELDS_MIN = "utilization.gpu,clocks.sm,power.draw,temperature.gpu"

    def __init__(self):
        self.samples, self.proc, self.ok = [], None, False
        for f in (self.FIELDS_FULL, self.FIELDS_OLD, self.FIELDS_MIN):
            try:
                r = subprocess.run(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10)
            except Exception:
                return
            if r.returncode == 0 and "Field" not in r.stdout:
                self.fields = f; self.ok = True
                break
        if not self.ok:
            return
        self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={self.fields}", "--format=csv,noheader,nounits",
                                      "-lms", "500"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            try:
                util, clk, pw, temp = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                reasons = int(parts[4], 16) if len(parts) > 4 and parts[4].startswith("0x") else 0
            except (ValueError, IndexError):
                continue
            self.samples.append((time.time(), util, clk, pw, temp, reasons))

    def window(self, t0, t1):
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return "no nvidia-smi samples"
        n = len(s)
        reasons = 0
        for x in s:
            reasons |= x[5]
        why = ",".join(v for k, v in THROTTLE_BITS.items() if reasons & k) or "none"
        return (f"util {sum(x[1] for x in s) / n:3.0f}%  SM {sum(x[2] for x in s) / n:4.0f} MHz  "
                f"{sum(x[3] for x in s) / n:3.0f} W  {max(x[4] for x in s):2.0f} C  limits: {why}")

    def stop(self):
        if self.proc:
            self.proc.terminate()


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------- steps
class BareStep:
    """G0: forward/backward/optimizer only (no clipping, no scheduler); targets left on CPU as in old stage 3."""

    def __init__(self, model, opt, scaler, dev, amp):
        self.m, self.o, self.s, self.dev, self.amp = model, opt, scaler, dev, amp
        self.ctc = nn.CTCLoss(blank=cs.BLANK, zero_infinity=True)

    def __call__(self, b):
        x = b["images"].to(self.dev, non_blocking=True)
        with torch.autocast(self.dev.type, enabled=self.amp, dtype=torch.float16):
            logits, lengths = self.m(x, b["widths"])
        loss = self.ctc(logits.float().log_softmax(-1), b["targets"], lengths, b["target_lengths"])
        self.o.zero_grad(set_to_none=True)
        self.s.scale(loss).backward(); self.s.step(self.o); self.s.update()
        return loss.detach()


class OldTrainerStep:
    """G1: the previous src/train.py step, line for line (isfinite check, clip, 2x get_scale, conditional
    scheduler, loss.item())."""

    def __init__(self, model, opt, sched, scaler, dev, amp, clip):
        self.m, self.o, self.sch, self.s, self.dev, self.amp, self.clip = model, opt, sched, scaler, dev, amp, clip
        self.ctc = nn.CTCLoss(blank=cs.BLANK, zero_infinity=True)

    def __call__(self, b):
        x = b["images"].to(self.dev, non_blocking=True)
        with torch.autocast(self.dev.type, enabled=self.amp, dtype=torch.float16):
            logits, lengths = self.m(x, b["widths"])
        loss = self.ctc(logits.float().log_softmax(-1), b["targets"], lengths, b["target_lengths"])
        if not torch.isfinite(loss):
            self.o.zero_grad(set_to_none=True); return loss.detach()
        self.o.zero_grad(set_to_none=True)
        self.s.scale(loss).backward()
        self.s.unscale_(self.o)
        nn.utils.clip_grad_norm_(self.m.parameters(), self.clip)
        before = self.s.get_scale()
        self.s.step(self.o); self.s.update()
        if not self.amp or self.s.get_scale() >= before:
            self.sch.step()
        loss.item()
        return loss.detach()


def make_step(kind, tc, mc, dev, amp, fused=False, channels_last=False, total_steps=100_000):
    tc2 = dict(tc); tc2["fused_adamw"] = fused
    model = build_model(mc, dev, channels_last).train()
    opt, sched = build_optim(model, tc2, total_steps, dev)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    clip = float(tc.get("grad_clip", 5.0))
    if kind == "bare":
        return BareStep(model, opt, scaler, dev, amp)
    if kind == "old":
        return OldTrainerStep(model, opt, sched, scaler, dev, amp, clip)
    return TrainStep(model, opt, sched, scaler, dev, amp, clip, channels_last)


def run_cached(step, batches, steps, dev, mon, label, bs):
    for b in batches[:3]:                         # warm-up: CUDA init, cuDNN heuristics, allocator
        step(b)
    sync(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(steps):
        step(batches[i % len(batches)])
    sync(dev)
    t1 = time.time()
    ips = steps * bs / (t1 - t0)
    mem = torch.cuda.max_memory_reserved() / 2**30 if dev.type == "cuda" else 0.0
    print(f"  {label:<44} {steps / (t1 - t0):5.2f} batch/s {ips:6.0f} img/s  mem {mem:4.2f} GB | {mon.window(t0, t1)}",
          flush=True)
    return ips, mem


def collect(ds, bs, n, workers=4):
    it = iter(make_loader(ds, bs, workers))
    out = []
    while len(out) < n:
        b = next(it)
        if b is not None:
            out.append(b)
    del it
    return out


def loop_test(label, ds, sampler, bs, workers, tc, mc, dev, amp, fused, cl, steps, mon):
    """The real training loop for `steps` timed steps (10 warm-up steps excluded)."""
    if dev.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        retries0 = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    try:
        step = make_step("new", tc, mc, dev, amp, fused, cl)
        it = iter(make_loader(ds, bs, workers, sampler=sampler))
        wait = 0.0; imgs = 0; t0 = None; pad_cols = true_cols = 0; nb = 0
        for s in range(steps + 10):
            tw = time.time()
            try:
                b = next(it)
            except StopIteration:                  # tiny epoch: start another pass
                it = iter(make_loader(ds, bs, workers, sampler=sampler)); b = next(it)
            tw = time.time() - tw
            if b is None:
                continue
            loss = step(b)
            if s == 9:
                loss.item(); t0 = time.time(); wait = 0.0; imgs = 0
            elif s > 9:
                n = b["images"].shape[0]
                wait += tw; imgs += n; nb += 1
                pad_cols += n * b["images"].shape[3]; true_cols += int(b["widths"].sum())
        loss.item(); t1 = time.time()
    except torch.cuda.OutOfMemoryError:
        print(f"  {label:<34} OUT OF GPU MEMORY", flush=True)
        return 0.0
    ips = imgs / (t1 - t0)
    mem = retr = ""
    if dev.type == "cuda":
        mem = f"mem {torch.cuda.max_memory_reserved() / 2**30:4.1f} GB"
        retr = f"retries {torch.cuda.memory_stats().get('num_alloc_retries', 0) - retries0}"
    print(f"  {label:<34} {ips:6.0f} img/s  wait {100 * wait / (t1 - t0):3.0f}%  {mem} {retr}  "
          f"padding {100 * (1 - true_cols / max(1, pad_cols)):3.0f}%  {imgs / max(1, nb):4.1f} img/batch | "
          f"{mon.window(t0, t1)}", flush=True)
    del it, step
    return ips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=None, help="default: config num_workers")
    ap.add_argument("--steps", type=int, default=80, help="timed steps per loop test")
    ap.add_argument("--gpu-tests", action="store_true", help="also run G0-G4 on cached batches")
    ap.add_argument("--val-items", type=int, default=3000)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    tc, mc = cfg["train"], cfg.get("model", {})
    bs = int(tc["batch_size"])
    workers = a.workers if a.workers is not None else int(tc.get("num_workers", 4))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(tc.get("amp", True)) and dev.type == "cuda"
    torch.backends.cudnn.benchmark = bool(tc.get("cudnn_benchmark", False))
    fused, cl = bool(tc.get("fused_adamw", False)), bool(tc.get("channels_last", False))
    cols = tc.get("max_batch_columns") or bs * 576
    pool = int(tc.get("bucket_pool", 50))
    print(f"device {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'})  amp={amp}  batch={bs}  "
          f"workers={workers}  fused_adamw={fused}  channels_last={cl}  cudnn.benchmark={torch.backends.cudnn.benchmark}")
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory: {total / 2**30:.1f} GB total, {free / 2**30:.1f} GB free before this process allocates")
    mon = GpuMonitor()
    print("nvidia-smi sampling:", "on" if mon.ok else "unavailable")

    sources = [build_source(s, cfg.get("seed", 42) + i) for i, s in enumerate(cfg["sources"])]
    ds = MixtureDataset(sources, [float(s.get("weight", 1)) for s in cfg["sources"]], int(tc["epoch_size"]),
                        seed=cfg.get("seed", 42))
    t = time.time()
    bucket = WidthBucketSampler(ds, bs, pool_batches=pool, max_columns=cols, seed=cfg.get("seed", 42))
    plan = bucket.batches(0)
    w = np.array([ds.est_width(i) or 0 for i in range(len(ds))])
    print(f"width plan in {time.time() - t:.0f} s: estimated input widths p50 {np.percentile(w, 50):.0f}, "
          f"p90 {np.percentile(w, 90):.0f}, p99 {np.percentile(w, 99):.0f}, max {w.max():.0f} px; "
          f"bucketed epoch = {len(plan)} batches (random: {math.ceil(len(ds) / bs)})")

    if a.gpu_tests:
        print(f"\n=== GPU only, 20 cached random batches, {a.steps} steps each")
        batches = collect(ds, bs, 20, workers)
        for key, label, kind, f_, c_ in (("G0", "bare step", "bare", False, False),
                                         ("G1", "OLD trainer step", "old", False, False),
                                         ("G2", "TrainStep, AdamW", "new", False, False),
                                         ("G3", "TrainStep, fused AdamW", "new", True, False),
                                         ("G4", "TrainStep, fused, channels_last", "new", True, True)):
            try:
                step = make_step(kind, tc, mc, dev, amp, f_, c_)
                run_cached(step, batches, a.steps, dev, mon, f"{key} {label}", bs)
            except torch.cuda.OutOfMemoryError:
                print(f"  {key} {label}: OUT OF MEMORY")
            del step
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        del batches

    print(f"\n=== full training loop, {a.steps} timed steps each (config options: fused_adamw={fused}, channels_last={cl})")
    r = {}
    r["random64"] = loop_test(f"random batches, batch {bs}", ds, None, bs, workers, tc, mc, dev, amp, fused, cl, a.steps, mon)
    r["bucket64"] = loop_test(f"width-bucketed, batch <= {bs}", ds, bucket, bs, workers, tc, mc, dev, amp, fused, cl,
                              a.steps, mon)
    r["random32"] = loop_test(f"random batches, batch {bs // 2} (memory probe)", ds, None, bs // 2, workers, tc, mc,
                              dev, amp, fused, cl, a.steps, mon)
    if r["random64"] and r["random32"] and r["random32"] > 1.5 * r["random64"]:
        print("  -> halving the batch made each image much cheaper: batch-64 random batches run out of GPU memory headroom")

    print("\n=== validation speed (full set is extrapolated)")
    vspec = cfg["val"][0]
    full = ManifestDataset(ROOT / vspec["path"], vspec["name"], where=vspec.get("where"))
    sub = ManifestDataset(ROOT / vspec["path"], vspec["name"], max_items=a.val_items, where=vspec.get("where"))
    model = build_model(mc, dev, cl)
    val_min = {}
    for name, smp in (("random order", None),
                      ("width-sorted", WidthBucketSampler(sub, bs, max_columns=cols, shuffle=False))):
        t = time.time()
        evaluate(model, make_loader(sub, bs, workers, sampler=smp), dev, amp)
        tv = time.time() - t
        val_min[name] = tv * len(full) / max(1, len(sub)) / 60
        print(f"  {name:<13}: {len(sub)} images in {tv:5.1f} s -> full validation ({len(full)}) ~ {val_min[name]:.1f} min")

    print(f"\n=== estimates ({int(tc['epochs'])} epochs x {int(tc['epoch_size'])} images)")
    for key, label, vk in (("random64", "random batches", "random order"), ("bucket64", "width-bucketed", "width-sorted")):
        if r.get(key):
            ep = int(tc["epoch_size"]) / r[key] / 60
            tot = int(tc["epochs"]) * (ep + val_min[vk] + 0.3)
            print(f"  {label:<15}: {r[key]:5.0f} img/s -> epoch {ep:4.1f} min + val {val_min[vk]:.1f} min; "
                  f"{int(tc['epochs'])} epochs ~ {tot / 60:.1f} h")
    print(f"  (config max_hours = {tc.get('max_hours')})")
    mon.stop()


if __name__ == "__main__":   # required on Windows
    main()
