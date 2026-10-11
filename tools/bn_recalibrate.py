"""
D0 diagnostic: BatchNorm recalibration. NO TRAINING, NO WEIGHT UPDATES.

Only the BatchNorm running_mean/running_var buffers are re-estimated, by forward passes under no_grad.
Every learned weight stays bit-identical; the source checkpoint is opened read-only and its SHA-256 is
verified before and after. Recalibrated copies are written to a NEW directory only.

  .venv\\Scripts\\python tools\\bn_recalibrate.py ^
      --ckpt runs\\experiment_001\\best.pth --config configs\\experiment_001.yaml ^
      --out-dir runs\\exp001_bn_d0

Arms (pre-declared):
  0 as-is            no recalibration (reproduces the checkpoint's own validation numbers)
  1 hw-clean         handwriting training images, NO augmentation
  2 hw-aug           handwriting training images, RealAugment("hw") as used in training
  3 synth            the config's synthetic source
  4 mixed            the config's full training stream (both sources at their configured weights)

Evaluation uses src.train.evaluate (the trainer's own code) on the config's `val` entries.
The test split is refused: this diagnostic must not touch it.
"""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import charset as cs  # noqa: E402
from src.augment import RealAugment  # noqa: E402
from src.data.datasets import ManifestDataset, MixtureDataset  # noqa: E402
from src.model.crnn_v2 import CRNNv2  # noqa: E402
from src.train import build_source, evaluate, make_loader, make_sampler  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_fingerprint(sd):
    """Hash of every non-BN-buffer tensor: proves the learned weights did not change."""
    h = hashlib.sha256()
    for k in sorted(sd):
        if k.endswith("running_mean") or k.endswith("running_var") or k.endswith("num_batches_tracked"):
            continue
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def bn_snapshot(model):
    return [(m.running_mean.detach().clone(), m.running_var.detach().clone()) for m in bn_layers(model)]


def bn_drift(model, before):
    """Mean relative change of the BN buffers, as evidence of how far recalibration moved them."""
    dm, dv = [], []
    for m, (mu0, var0) in zip(bn_layers(model), before):
        scale = var0.sqrt().clamp(min=1e-5)
        dm.append(((m.running_mean - mu0).abs() / scale).mean().item())
        dv.append((m.running_var / var0.clamp(min=1e-8)).log().abs().mean().item())
    return float(np.mean(dm)), float(np.mean(dv))


def build_model_from_ckpt(ck, device):
    margs = dict(ck["model_args"])
    margs["channels"] = tuple(margs["channels"])
    model = CRNNv2(**margs).to(device)
    model.load_state_dict(ck["model"])
    return model


def arm_dataset(arm, cfg, args, seed):
    """Datasets for recalibration. Each returns a MixtureDataset so the trainer's collate path is used."""
    src_specs = cfg["sources"]
    hw_spec = next(s for s in src_specs if s["type"] == "manifest")
    if arm in ("hw-clean", "hw-aug"):
        ds = ManifestDataset(ROOT / hw_spec["path"], hw_spec["name"],
                             augment=RealAugment(hw_spec["augment"]) if arm == "hw-aug" else None)
        return MixtureDataset([ds], [1.0], args.batches * args.batch_size, seed=seed)
    if arm == "synth":
        specs = [s for s in src_specs if s["type"] == "synth"]
        sources = [build_source(s, cfg.get("seed", 42) + 1 + i) for i, s in enumerate(specs)]
        return MixtureDataset(sources, [float(s.get("weight", 1)) for s in specs],
                              args.batches * args.batch_size, seed=seed)
    if arm == "mixed":
        sources = [build_source(s, cfg.get("seed", 42) + i) for i, s in enumerate(src_specs)]
        return MixtureDataset(sources, [float(s.get("weight", 1)) for s in src_specs],
                              args.batches * args.batch_size, seed=seed)
    raise ValueError(arm)


@torch.no_grad()
def recalibrate(model, ds, args, device, amp):
    """Re-estimate BN running statistics. No gradients, no optimiser, no weight change."""
    for m in bn_layers(model):
        m.reset_running_stats()
        m.momentum = None                     # cumulative moving average = exact mean over the batches seen
    model.train()
    loader = make_loader(ds, args.batch_size, args.workers)
    n_b = n_img = 0
    src_count = {}
    for batch in loader:
        if batch is None:
            continue
        x = batch["images"].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp, dtype=torch.float16):
            model(x, batch["widths"])
        n_b += 1; n_img += x.shape[0]
        for s in batch["sources"]:
            src_count[s] = src_count.get(s, 0) + 1
        if n_b >= args.batches:
            break
    model.eval()
    return n_b, n_img, src_count


def score(model, cfg, args, device, amp):
    out = {}
    for v in cfg.get("val", []):
        p = str(v["path"])
        if "test" in Path(p).name.lower():
            sys.exit(f"refusing to evaluate on {p}: D0 must not touch the test split")
        ds = ManifestDataset(ROOT / p, v["name"], where=v.get("where"), max_items=args.max_items)
        try:                                   # the project's make_sampler may carry extra keys (unknown_width_estimate)
            smp = make_sampler(ds, cfg["train"], train=False)
        except TypeError:
            smp = None
        summ, _ = evaluate(model, make_loader(ds, args.batch_size, args.eval_workers, sampler=smp), device, amp)
        a = summ["all"]
        out[v["name"]] = {"n": a["n"], "cer": a["cer"], "wer": a["wer"], "len_ratio": a["len_ratio"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/experiment_001/best.pth")
    ap.add_argument("--config", default="configs/experiment_001.yaml")
    ap.add_argument("--out-dir", default="runs/exp001_bn_d0")
    ap.add_argument("--arms", nargs="+", default=["as-is", "hw-clean", "hw-aug", "synth", "mixed"])
    ap.add_argument("--batches", type=int, default=200, help="recalibration batches per arm (200 x 64 = 12,800 images)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0, help="recalibration loader workers (0 = in-process)")
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--max-items", type=int, default=None, help="debug: cap validation size")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--save", action="store_true", help="write each recalibrated copy into --out-dir")
    a = ap.parse_args()

    ckpt = ROOT / a.ckpt
    before_hash = sha256(ckpt)
    cfg = yaml.safe_load(open(ROOT / a.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    out_dir = ROOT / a.out_dir

    print(f"D0 BatchNorm recalibration diagnostic — NO TRAINING")
    print(f"  checkpoint {ckpt}")
    print(f"  sha256 before {before_hash}")
    print(f"  config     {a.config}   device {device}  amp {amp}")
    print(f"  recalibration: {a.batches} batches x {a.batch_size} = {a.batches * a.batch_size:,} images per arm, "
          f"seed {a.seed}")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    print(f"  checkpoint epoch {ck.get('epoch')}, classes {ck['model_args']['num_classes']}")
    if list(ck.get("charset", cs.CHARSET)) != list(cs.CHARSET):
        sys.exit("checkpoint charset differs from src/charset.py")
    ref_weights = weight_fingerprint(ck["model"])

    results, rows = {}, []
    for arm in a.arms:
        t0 = time.time()
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        model = build_model_from_ckpt(copy.deepcopy(ck), device)
        before_bn = bn_snapshot(model)
        info = ""
        if arm != "as-is":
            ds = arm_dataset(arm, cfg, a, a.seed)
            n_b, n_img, src_count = recalibrate(model, ds, a, device, amp)
            dm, dv = bn_drift(model, before_bn)
            mix = ", ".join(f"{k} {100 * v / max(1, sum(src_count.values())):.0f}%" for k, v in sorted(src_count.items()))
            info = f"{n_b} batches / {n_img:,} images ({mix}); BN drift: mean {dm:.3f} sd, var log-ratio {dv:.3f}"
        else:
            model.eval()
        fp = weight_fingerprint(model.state_dict())
        assert fp == ref_weights, "learned weights changed — this must never happen"
        res = score(model, cfg, a, device, amp)
        results[arm] = {"eval": res, "info": info, "weights_unchanged": True}
        rows.append((arm, res))
        print(f"\n[{arm}] {time.time() - t0:.0f}s  {info}")
        print(f"   weights unchanged: yes (fingerprint {fp[:16]}…)")
        for name, m in res.items():
            print(f"   {name:<18} n={m['n']:>6}  CER {100 * m['cer']:7.4f}%  WER {100 * m['wer']:7.4f}%  "
                  f"len-ratio {m['len_ratio']:.4f}")
        if a.save and arm != "as-is":
            out_dir.mkdir(parents=True, exist_ok=True)
            ck2 = copy.deepcopy(ck)
            ck2["model"] = {k: v.cpu() for k, v in model.state_dict().items()}
            ck2["bn_recalibrated"] = {"arm": arm, "batches": a.batches, "batch_size": a.batch_size,
                                      "seed": a.seed, "source_checkpoint": str(a.ckpt), "source_sha256": before_hash}
            torch.save(ck2, out_dir / f"bn_{arm}.pth")
            print(f"   saved {out_dir / f'bn_{arm}.pth'}")

    print("\n=== summary (validation only; the test split was not used)")
    names = list(rows[0][1].keys())
    print(f"  {'arm':<10}" + "".join(f"{n:>26}" for n in names))
    for arm, res in rows:
        print(f"  {arm:<10}" + "".join(f"{100 * res[n]['cer']:>13.4f}% {100 * res[n]['wer']:>10.4f}%" for n in names))
    print("  (each cell: CER then WER)")

    after_hash = sha256(ckpt)
    print(f"\n  sha256 after  {after_hash}")
    print(f"  source checkpoint unchanged: {'YES' if after_hash == before_hash else 'NO — STOP'}")
    if a.save:
        (out_dir / "d0_results.json").write_text(json.dumps(
            {"source": str(a.ckpt), "sha256": before_hash, "config": a.config, "batches": a.batches,
             "batch_size": a.batch_size, "seed": a.seed, "arms": results}, indent=1), encoding="utf-8")
        print(f"  wrote {out_dir / 'd0_results.json'}")


if __name__ == "__main__":   # required on Windows
    main()
