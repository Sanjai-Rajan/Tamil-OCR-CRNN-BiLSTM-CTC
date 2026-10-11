"""
Experiment-001 prerequisite audit. READ-ONLY: writes no file, creates no run directory, no checkpoint,
trains nothing. Everything it reports is measured in-process from the existing code and data.

  .venv\\Scripts\\python tools\\bench_synth.py --vocab data\\text_corpus\\tamil_correction_vocabulary.json

Sections
  A  required paths exist
  B  generator/text-sampler configuration actually in effect (including the silent-vocab-fallback check)
  C  synthetic generation throughput and the width distribution of synthetic images at model input height
  D  WidthBucketSampler safety: estimated vs actual batch width for sources with unknown width
  E  full training loop on a 50/50 handwriting+synthetic mixture (throughput, GPU memory, padding)
  F  seed-collision audit against the frozen synthetic validation set
  G  cost estimate from the measured numbers
  H  (--config, runs alone) the training loop built from a config exactly as src/train.py builds it, with width
     bucketing and unknown_width_estimate: throughput, memory per window, padding, batch columns, stress probe,
     validation speed and the resulting runtime estimate

  .venv\\Scripts\\python tools\\bench_synth.py --config configs\\experiment_001.yaml --steps 400
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import preprocess as pp  # noqa: E402
from src.data.datasets import MixtureDataset, SynthSource, WidthBucketSampler  # noqa: E402
from src.synth.generator import SynthWordGenerator  # noqa: E402
from src.synth.text_sampler import TextSampler  # noqa: E402
from src.train import TrainStep, build_model, build_optim, build_source, make_loader  # noqa: E402

FALLBACK_WORDS = 12          # size of TextSampler's built-in list when no vocab is loaded


def section(t):
    print(f"\n=== {t}", flush=True)


def loop_section(a, bs, dev_holder):
    """Section E. Runs BEFORE any rendering happens in this process: on Linux (fork start method)
    rendering a synthetic sample in the parent and then starting DataLoader workers deadlocks
    (FreeType/HarfBuzz state is not fork-safe). Windows uses spawn and is not affected."""
    section(f"E  full training loop, 50/50 handwriting + synthetic, {a.steps} timed steps")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    cfg_sources = [
        {"name": "iiit_hw", "type": "manifest", "path": a.train_manifest, "augment": "hw", "weight": 0.5},
        {"name": "synth_modern", "type": "synth", "vocab": a.vocab, "old_print_prob": 0.0, "weight": 0.25,
         "length": 1_000_000},
        {"name": "synth_old", "type": "synth", "vocab": a.vocab, "old_print_prob": 1.0, "weight": 0.25,
         "length": 1_000_000},
    ]
    tc = {"batch_size": bs, "lr": 0.001, "weight_decay": 0.01, "warmup_pct": 0.05, "grad_clip": 5.0,
          "fused_adamw": True, "amp": True}
    sources = [build_source(s, a.seed + i) for i, s in enumerate(cfg_sources)]
    for s, spec in zip(sources, cfg_sources):
        print(f"  source {spec['name']:<14} type={spec['type']:<8} weight={spec['weight']} len={len(s):,}")
    ds = MixtureDataset(sources, [s["weight"] for s in cfg_sources], a.epoch_size, seed=a.seed)
    model = build_model({}, dev).train()
    opt, sched = build_optim(model, tc, 100_000, dev)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step = TrainStep(model, opt, sched, scaler, dev, amp, 5.0, False)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        retries0 = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    it = iter(make_loader(ds, bs, a.workers))
    wait = 0.0; imgs = 0; t0 = None; pad = true = 0; nb = 0; src_count = {}
    try:
        for s in range(a.steps + 10):
            tw = time.time(); b = next(it); tw = time.time() - tw
            if b is None:
                continue
            loss = step(b)
            if s == 9:
                loss.item(); t0 = time.time(); wait = 0.0; imgs = 0
            elif s > 9:
                n = b["images"].shape[0]
                wait += tw; imgs += n; nb += 1
                pad += n * b["images"].shape[3]; true += int(b["widths"].sum())
                for nm in b["sources"]:
                    src_count[nm] = src_count.get(nm, 0) + 1
        loss.item(); t1 = time.time()
    except torch.cuda.OutOfMemoryError:
        print("  OUT OF GPU MEMORY during the mixed loop")
        return None
    ips = imgs / (t1 - t0)
    print(f"  {ips:.0f} img/s   waiting for data {100 * wait / (t1 - t0):.0f}%   "
          f"padding {100 * (1 - true / max(1, pad)):.0f}%   {imgs / max(1, nb):.1f} img/batch")
    if dev.type == "cuda":
        print(f"  peak GPU memory {torch.cuda.max_memory_reserved() / 2**30:.2f} GB, allocator retries "
              f"{torch.cuda.memory_stats().get('num_alloc_retries', 0) - retries0}")
    tot = sum(src_count.values())
    print("  observed source mix: " + ", ".join(f"{k} {100 * v / tot:.1f}%" for k, v in sorted(src_count.items())))

    return ips


def config_loop_section(a):
    """Section H: the Experiment-001 training loop exactly as src/train.py builds it from a config (sources,
    MixtureDataset, WidthBucketSampler incl. unknown_width_estimate, make_loader, model, optimiser, TrainStep),
    for epoch 0's batch order. Trains an in-memory model only: no run directory, no checkpoint, no file."""
    import subprocess
    import yaml
    from src.data.datasets import PAD_W_MULTIPLE
    from src.train import evaluate, make_sampler, seed_everything, build_val

    cfg = yaml.safe_load(open(ROOT / a.config, encoding="utf-8"))
    tc, mc = cfg["train"], cfg.get("model", {})
    section(f"H  bucketed training loop from {a.config}: {a.steps} timed steps (+10 warm-up), then stress probe")
    seed_everything(cfg.get("seed", 42))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(tc.get("amp", True)) and dev.type == "cuda"
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = bool(tc.get("cudnn_benchmark", False))
    bs, workers = int(tc["batch_size"]), int(tc.get("num_workers", 4))
    cap, unk = tc.get("max_batch_columns"), tc.get("unknown_width_estimate")
    sources = [build_source(s, cfg.get("seed", 42) + i) for i, s in enumerate(cfg["sources"])]
    weights = [float(s.get("weight", 1.0)) for s in cfg["sources"]]
    for s, spec in zip(sources, cfg["sources"]):
        extra = f" vocab words {len(s.gen.sampler.words):,} old_print_prob {s.gen.old_print_prob}" if hasattr(s, "gen") else ""
        print(f"  source {spec['name']:<12} type={spec['type']:<8} weight={spec.get('weight', 1.0)} len={len(s):,}{extra}")
    ds = MixtureDataset(sources, weights, int(tc["epoch_size"]), seed=cfg.get("seed", 42))
    t = time.time()
    sampler = make_sampler(ds, tc, train=True, seed=cfg.get("seed", 42))
    sampler.set_epoch(0); ds.set_epoch(0)
    plan = sampler.batches(0)
    # the estimates the plan was cut from (PairedSourceBatchSampler keeps them in _widths_and_groups)
    est_w = sampler._widths() if hasattr(sampler, "_widths") else sampler._widths_and_groups(0)[0]
    import hashlib
    print(f"  sampler {type(sampler).__name__}; epoch-0 plan sha256 "
          f"{hashlib.sha256(json.dumps(plan).encode()).hexdigest()[:16]}... (compare with the CPU preflight)")
    rw = lambda w: -(-int(w) // PAD_W_MULTIPLE) * PAD_W_MULTIPLE  # noqa: E731
    est_cols = np.array([len(b) * max(rw(est_w[i]) for i in b) for b in plan])
    is_syn = np.array([ds.locate(i)[0] > 0 for i in range(len(ds))]) if len(sources) > 1 else None
    kinds = {"synthetic-only": 0, "handwriting-only": 0, "mixed": 0}
    for b in plan:
        s_ = is_syn[b].sum()
        kinds["synthetic-only" if s_ == len(b) else "handwriting-only" if s_ == 0 else "mixed"] += 1
    print(f"  epoch-0 plan: {len(plan):,} batches (plain batching would be {math.ceil(len(ds) / bs):,}), "
          f"planned in {time.time() - t:.0f} s; max_batch_columns={cap}, unknown_width_estimate={unk}")
    print(f"  plan batch kinds: " + ", ".join(f"{k} {v:,} ({100 * v / len(plan):.1f}%)" for k, v in kinds.items()))
    syn_sizes = [len(b) for b in plan if is_syn[b].all()]
    if syn_sizes:
        print(f"  synthetic-only batch size: min {min(syn_sizes)}, median {int(np.median(syn_sizes))}, max {max(syn_sizes)}")
    print(f"  ESTIMATED columns per batch: median {int(np.median(est_cols)):,}, max {est_cols.max():,}, "
          f"over cap {int((est_cols > cap).sum()) if cap else 0}")

    model = build_model(mc, dev)
    total_steps = len(plan) * int(tc["epochs"])                # train.py sums the exact per-epoch counts
    opt, sched = build_optim(model, tc, total_steps, dev)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step = TrainStep(model, opt, sched, scaler, dev, amp, float(tc.get("grad_clip", 5.0)))
    vram = torch.cuda.mem_get_info()[1] if dev.type == "cuda" else 0

    def smi_mem():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            u, tot = (int(x) for x in r.stdout.strip().split(","))
            return f"nvidia-smi {u:,}/{tot:,} MiB"
        except Exception:
            return "nvidia-smi n/a"

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        retries0 = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    t_create = time.time()
    it = iter(make_loader(ds, bs, workers, sampler=sampler))
    rows, windows = [], []
    t0 = win_t = None; win_imgs = win_wait = 0; max_wait = 0.0; first_batch = None
    try:
        for s in range(a.steps + 10):
            tw = time.time(); b = next(it); tw = time.time() - tw
            if s == 0:
                first_batch = time.time() - t_create
            if b is None:
                continue
            loss = step(b)
            n, W = b["images"].shape[0], b["images"].shape[3]
            syn = sum(nm != cfg["sources"][0]["name"] for nm in b["sources"])
            rows.append({"n": n, "cols": n * W, "true": int(b["widths"].sum()), "syn": syn, "wait": tw,
                         "est": int(est_cols[s]), "timed": s >= 10})
            if s == 9:
                loss.item(); t0 = win_t = time.time(); win_imgs = win_wait = 0
            elif s > 9:
                win_imgs += n; win_wait += tw; max_wait = max(max_wait, tw)
                if (s - 9) % 50 == 0:
                    loss.item(); now = time.time()
                    windows.append((s - 9, win_imgs / (now - win_t), 100 * win_wait / (now - win_t),
                                    torch.cuda.max_memory_reserved() / 2**30 if dev.type == "cuda" else 0,
                                    torch.cuda.memory_reserved() / 2**30 if dev.type == "cuda" else 0, smi_mem()))
                    win_t, win_imgs, win_wait = now, 0, 0
        loss.item(); t1 = time.time()
    except torch.cuda.OutOfMemoryError:
        print("  OUT OF GPU MEMORY during the loop")
        return None
    T = [r for r in rows if r["timed"]]
    imgs = sum(r["n"] for r in T); ips = imgs / (t1 - t0)
    cols = np.array([r["cols"] for r in T]); est = np.array([r["est"] for r in T])
    pad = 100 * (1 - sum(r["true"] for r in T) / max(1, sum(r["cols"] for r in T)))
    print(f"  DataLoader: {workers} workers, first batch after {first_batch:.1f} s (worker start-up), "
          f"waiting for data {100 * sum(r['wait'] for r in T) / (t1 - t0):.1f}% of timed wall time, "
          f"longest single wait {max_wait:.2f} s")
    print(f"  timed: {len(T)} steps, {imgs:,} images in {t1 - t0:.0f} s -> {ips:.1f} img/s  "
          f"({imgs / len(T):.1f} img/batch), padding {pad:.1f}%")
    print("  per 50-step window: step  img/s  data-wait  peak-reserved  reserved  nvidia-smi")
    for w in windows:
        print(f"    {w[0]:5d}  {w[1]:6.1f}  {w[2]:5.1f}%  {w[3]:6.2f} GB  {w[4]:6.2f} GB  {w[5]}")
    by = {"synthetic-only": [], "handwriting-only": [], "mixed": []}
    for r in T:
        by["synthetic-only" if r["syn"] == r["n"] else "handwriting-only" if r["syn"] == 0 else "mixed"].append(r)
    for k, v in by.items():
        if v:
            c = np.array([r["cols"] for r in v])
            p = 100 * (1 - sum(r["true"] for r in v) / sum(r["cols"] for r in v))
            print(f"  {k:16}: {len(v):4d} batches, {np.mean([r['n'] for r in v]):.1f} img/batch, actual columns "
                  f"median {int(np.median(c)):,} max {c.max():,}, over cap {int((c > cap).sum()) if cap else 0}, padding {p:.1f}%")
    print(f"  actual vs estimated columns: actual > estimate in {int((cols > est).sum())} of {len(T)} batches; "
          f"largest actual {cols.max():,} ({100 * cols.max() / cap:.0f}% of cap)" if cap else "")
    if dev.type == "cuda":
        pr = torch.cuda.max_memory_reserved()
        print(f"  GPU memory: peak reserved {pr / 2**30:.2f} GB, peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
              f"device VRAM {vram / 2**30:.2f} GB, allocator retries {torch.cuda.memory_stats().get('num_alloc_retries', 0) - retries0}"
              f" -> {'ABOVE VRAM: spill into shared system memory likely' if pr > vram else 'within VRAM'}")
    loop_ips = ips
    del it                                                     # stop the loader's workers before the probe

    # stress probe: the heaviest planned batches of the whole epoch, run through the same TrainStep
    if a.stress_batches and dev.type == "cuda":
        from src.data.datasets import collate
        order = np.argsort(-est_cols)[:a.stress_batches]
        syn_idx = [k for k, b in enumerate(plan) if is_syn[b].all()]
        order = list(order) + syn_idx[:a.stress_batches]         # synthetic widths are unknown until rendered
        worst = 0; t = time.time()
        for k in order:
            items = [ds[i] for i in plan[k]]
            b = collate(items)
            if b is None:
                continue
            worst = max(worst, b["images"].shape[0] * b["images"].shape[3])
            step(b)
        torch.cuda.synchronize()
        pr = torch.cuda.max_memory_reserved()
        print(f"  stress probe: {len(order)} batches (the {a.stress_batches} largest by estimate + {min(len(syn_idx), a.stress_batches)} "
              f"synthetic-only) in {time.time() - t:.0f} s; largest actual columns {worst:,}; peak reserved now "
              f"{pr / 2**30:.2f} GB ({'ABOVE' if pr > vram else 'within'} {vram / 2**30:.2f} GB VRAM); {smi_mem()}")

    # replay of specific epoch-0 plan positions, e.g. the heaviest ACTUAL batches recorded by a CPU preflight
    if a.replay_batches and dev.type == "cuda":
        from src.data.datasets import collate
        ks = [int(x) for x in a.replay_batches.split(",") if x.strip()]
        ds.set_epoch(0)
        t = time.time(); worst = 0
        print(f"  replay: {len(ks)} recorded epoch-0 plan positions, same TrainStep")
        for k in ks:
            if not 0 <= k < len(plan):
                print(f"    k={k}: outside the epoch-0 plan ({len(plan)} batches) - skipped"); continue
            b = collate([ds[i] for i in plan[k]])
            if b is None:
                print(f"    k={k}: empty batch - skipped"); continue
            n, W = b["images"].shape[0], b["images"].shape[3]
            worst = max(worst, n * W)
            step(b); torch.cuda.synchronize()
            print(f"    k={k}: {n} images, padded width {W}, columns {n * W:,}, "
                  f"peak reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB")
        pr = torch.cuda.max_memory_reserved()
        print(f"  replay done in {time.time() - t:.0f} s; largest replayed columns {worst:,}; peak reserved "
              f"{pr / 2**30:.2f} GB ({'ABOVE' if pr > vram else 'within'} {vram / 2**30:.2f} GB VRAM); "
              f"allocator retries {torch.cuda.memory_stats().get('num_alloc_retries', 0) - retries0}; {smi_mem()}")

    # validation pass on the configured validation sets (train.py evaluate + width-sorted val sampler)
    vt = 0.0; vn = 0
    for v in cfg.get("val", []):
        vds = build_val(v)
        t = time.time()
        evaluate(model, make_loader(vds, bs, workers, sampler=make_sampler(vds, tc, train=False)), dev, amp)
        el = time.time() - t; vt += el; vn += len(vds)
        print(f"  validation {v['name']:<18} {len(vds):6,} images in {el:5.1f} s")
    ep_train = int(tc["epoch_size"]) / loop_ips / 60
    tot_h = int(tc["epochs"]) * (ep_train + vt / 60) / 60
    print(f"  -> per epoch: training {ep_train:.1f} min + validation {vt / 60:.1f} min ({vn:,} images); "
          f"{tc['epochs']} epochs = {tot_h:.1f} h (excludes per-epoch worker start-up and checkpoint writes); "
          f"max_hours {tc.get('max_hours')}")
    thr = int(tc["epochs"]) * int(tc["epoch_size"]) / ((float(tc.get("max_hours", 1e9)) * 3600) - int(tc["epochs"]) * vt)
    print(f"  throughput below which max_hours would cut the schedule: {thr:.1f} img/s; decision threshold 90 img/s: "
          f"{'MET' if loop_ips >= 90 else 'NOT MET'}")
    return loop_ips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="data/text_corpus/tamil_correction_vocabulary.json")
    ap.add_argument("--fonts", default="assets/fonts")
    ap.add_argument("--train-manifest", default="data/real/iiit_hw/manifest_train.jsonl")
    ap.add_argument("--synth-val", default="data/synth_val/manifest.jsonl")
    ap.add_argument("--samples", type=int, default=400, help="synthetic samples for C/D")
    ap.add_argument("--steps", type=int, default=60, help="timed steps for E")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-batch-columns", type=int, default=32768)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--epoch-size", type=int, default=151472, help="proposed epoch size (2x baseline for 50/50)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-seed-base", type=int, default=900_000_000)
    ap.add_argument("--val-old-offset", type=int, default=50_000_000)
    ap.add_argument("--val-n", type=int, default=3000, help="frozen val samples per style")
    ap.add_argument("--one-synth-source", action="store_true",
                    help="simulate one synth source with old_print_prob 0.5 (design review 6.2) instead of two sources at 0.0/1.0")
    ap.add_argument("--skip-loop", action="store_true", help="skip section E (no GPU work)")
    ap.add_argument("--config", default=None,
                    help="run ONLY section H: the bucketed training loop built from this config exactly as src/train.py does")
    ap.add_argument("--stress-batches", type=int, default=30,
                    help="section H: also run the N heaviest planned batches and N synthetic-only batches")
    ap.add_argument("--replay-batches", default=None,
                    help="section H: comma-separated epoch-0 plan positions to replay through TrainStep "
                         "(e.g. the heaviest actual batches listed by a CPU preflight)")
    a = ap.parse_args()
    if a.config:
        config_loop_section(a)
        return

    bs = a.batch_size

    # ------------------------------------------------------------------ A: paths
    section("A  required paths")
    paths = {"fonts dir": ROOT / a.fonts, "vocab json": ROOT / a.vocab,
             "IIIT train manifest": ROOT / a.train_manifest,
             "frozen synth val manifest": ROOT / a.synth_val}
    ok = True
    for k, p in paths.items():
        e = p.exists()
        ok &= e
        size = ""
        if e and p.is_file():
            size = f"{p.stat().st_size:,} bytes"
        elif e:
            size = f"{sum(1 for _ in p.rglob('*') if _.is_file())} files"
        print(f"  {'OK  ' if e else 'MISSING'} {k:28} {p}  {size}")
    if not ok:
        print("  -> missing paths: the remaining sections may fail or be meaningless")

    # ------------------------------------------------------------------ B: configuration in effect
    section("B  generator / text sampler actually in effect")
    vocab_path = str(ROOT / a.vocab)
    ts = TextSampler(vocab_path, seed=0)
    print(f"  vocabulary words loaded: {len(ts.words):,}")
    if len(ts.words) <= FALLBACK_WORDS:
        print(f"  *** WARNING: TextSampler is using its built-in {FALLBACK_WORDS}-word fallback list.")
        print("      A wrong or missing vocab path does NOT raise an error; training would see 12 words")
        print("      plus pseudo-words. Check the --vocab path before any synthetic training.")
    gen = SynthWordGenerator([str(ROOT / a.fonts)], vocab_json=vocab_path, old_print_prob=0.5, text_seed=a.seed)
    print(f"  font families: {len(gen.family_names)}  (book/serif: {len(gen.book_families)}, "
          f"traditional-capable: {len(gen.trad_families)})")
    print(f"  text mix (frequent, long-tail, pseudo, number, punctuated): {ts.mix}")
    print("  style selection: SynthWordGenerator.sample() draws old = rng.random() < old_print_prob as its")
    print("    FIRST random draw; old_print_prob comes from the source's 'old_print_prob' config key")
    print("    (src/train.py build_source, default 0.3). 50/50 is therefore either one source with")
    print("    old_print_prob: 0.5, or two sources with 0.0 and 1.0 and equal weights (exact ratio).")

    ips = None if a.skip_loop else loop_section(a, bs, None)
    # ------------------------------------------------------------------ C: generation throughput + widths
    section(f"C  synthetic generation: {a.samples} samples, single process")
    widths, styles, texts = [], [], []
    t0 = time.time()
    for i in range(a.samples):
        img, text, meta = gen.sample(1_000_000_000 + i)      # seeds outside train and val ranges
        h, w = img.shape[:2]
        widths.append(int(np.clip(round(w * pp.TARGET_H / max(1, h)), pp.MIN_W, pp.MAX_W)))
        styles.append(meta["style"]); texts.append(text)
    el = time.time() - t0
    widths = np.array(widths)
    print(f"  {a.samples / el:.0f} samples/s per process ({1000 * el / a.samples:.1f} ms/sample); "
          f"rejected (illegible, fell back to modern): {gen.rejected}")
    print(f"  style split: modern {styles.count('modern')}, old {styles.count('old')}")
    print(f"  input width at height {pp.TARGET_H}: p50 {np.percentile(widths, 50):.0f}, "
          f"p90 {np.percentile(widths, 90):.0f}, p99 {np.percentile(widths, 99):.0f}, max {widths.max()} px")
    print(f"  label length: p50 {np.percentile([len(t) for t in texts], 50):.0f}, "
          f"max {max(len(t) for t in texts)} chars")

    # ------------------------------------------------------------------ D: bucketing safety
    section("D  WidthBucketSampler safety with unknown-width (synthetic) samples")
    print("  SynthSource.est_width() returns None, so the sampler substitutes the MEDIAN of the known")
    print("  (real-image) widths for every synthetic sample. max_batch_columns is then enforced against")
    print("  that estimate, not against the real width.")
    worst = bs * int(widths.max())
    p99 = bs * int(np.percentile(widths, 99))
    print(f"  configured cap        : {a.max_batch_columns:,} columns (= {bs} x {a.max_batch_columns // bs} px)")
    print(f"  actual columns if one batch is all-synthetic: p99 {p99:,}, worst case {worst:,}")
    over = 100 * (worst / a.max_batch_columns - 1)
    print(f"  -> worst observed synthetic batch would exceed the cap by {over:.0f}% "
          f"({'RISK: memory is not bounded as intended' if worst > a.max_batch_columns else 'within cap'})")
    print(f"  safe cap for these samples: {worst:,} columns, or set the unknown-width estimate to "
          f"{int(np.percentile(widths, 99))}-{int(widths.max())} px instead of the real-image median")

    # ------------------------------------------------------------------ F: seed audit (cheap, before E)
    section("F  seed-collision audit against the frozen synthetic validation set")
    val_seeds = set(range(a.val_seed_base, a.val_seed_base + a.val_n)) | \
                set(range(a.val_seed_base + a.val_old_offset, a.val_seed_base + a.val_old_offset + a.val_n))
    print(f"  frozen validation seeds: {len(val_seeds):,} values "
          f"({a.val_seed_base}..+{a.val_n} modern, {a.val_seed_base + a.val_old_offset}..+{a.val_n} old)")
    mf = ROOT / a.synth_val
    if mf.exists():
        st = {}
        for line in open(mf, encoding="utf-8"):
            if line.strip():
                st[json.loads(line).get("style", "?")] = st.get(json.loads(line).get("style", "?"), 0) + 1
        print(f"  frozen manifest styles: {st}")
    # Exact simulation of the proposed mixture's draws: MixtureDataset.locate picks the source and a
    # RANDOM item index j, then SynthSource.get hashes (seed_base, epoch, j). Enumerating j in order would
    # give a different seed stream, so the real sequence is reproduced here (no image is generated).
    n_src = 2 if a.one_synth_source else 3
    seed_base = [(a.seed + k) * 7919 for k in range(1, n_src)]          # build_source: seed = cfg.seed + index
    w = [0.5] + [0.5 / (n_src - 1)] * (n_src - 1)
    cum = np.cumsum([x / sum(w) for x in w])
    lens = [75_736] + [1_000_000] * (n_src - 1)
    hits, hw_epoch, n_synth = [], [], 0
    t0 = time.time()
    for e in range(a.epochs):
        hw = 0
        for idx in range(a.epoch_size):
            rng = random.Random(hash((a.seed, e, idx)) & 0x7FFFFFFF)
            k = int(np.searchsorted(cum, rng.random(), side="right")); k = min(k, n_src - 1)
            j = rng.randrange(lens[k])
            if k == 0:
                hw += 1
                continue
            n_synth += 1
            sd = hash((seed_base[k - 1], e, j)) & 0x7FFFFFFF
            if sd in val_seeds:
                hits.append((e, k, sd))
        hw_epoch.append(hw)
    hw_epoch = np.array(hw_epoch)
    print(f"  simulated {a.epochs} x {a.epoch_size:,} draws in {time.time() - t0:.0f} s "
          f"({'one synth source, old_print_prob 0.5' if a.one_synth_source else 'two synth sources, 0.0 and 1.0'})")
    print(f"  synthetic draws: {n_synth:,};  expected collisions by chance: "
          f"{n_synth * len(val_seeds) / 2**31:.2f}")
    print(f"  EXACT seed collisions with the frozen validation set: {len(hits)}")
    identical = 0
    for e, k, sd in hits:
        val_old = sd >= a.val_seed_base + a.val_old_offset
        p_old = 0.5 if a.one_synth_source else (0.0 if k == 1 else 1.0)
        train_old = random.Random(sd).random() < p_old        # sample()'s first draw decides the style
        same = (train_old == val_old)
        identical += same
        print(f"    epoch {e:2d}  seed {sd}  validation style {'old' if val_old else 'modern':6}  "
              f"training style {'old' if train_old else 'modern':6}  -> "
              f"{'IDENTICAL image' if same else 'different image'}")
    print(f"  collisions that would reproduce a validation image exactly: {identical} "
          f"({100 * identical / 6000:.3f}% of the 6,000-image set), assuming the same vocabulary file and fonts")
    print(f"  handwriting draws per epoch: mean {hw_epoch.mean():,.0f}, sd {hw_epoch.std():.0f}, "
          f"min {hw_epoch.min():,}, max {hw_epoch.max():,} (baseline 75,736)")
    print(f"  handwriting draws over the run: {hw_epoch.sum():,} (baseline {a.epochs * 75736:,}, "
          f"{100 * hw_epoch.sum() / (a.epochs * 75736):.2f}%)")
    print("  a colliding seed reproduces the validation IMAGE only if the style coin-flip also matches")
    print("  (validation modern used old_print_prob=0.0, old used 1.0; training would use 0.5 -> ~50%),")
    print("  and only if the same vocabulary file is used. A different vocab makes collisions impossible.")

    # ------------------------------------------------------------------ G: cost
    section("G  cost estimate from the measured loop throughput")
    if not ips:
        print("  section E was skipped or failed - no measured throughput to base an estimate on")
        return
    ep_min = a.epoch_size / ips / 60
    print(f"  epoch of {a.epoch_size:,} samples: {ep_min:.1f} min of training (validation excluded)")
    print(f"  {a.epochs} epochs: {a.epochs * ep_min / 60:.1f} h + validation time")
    print("  handwriting exposure per epoch = epoch_size x weight(iiit_hw) = "
          f"{a.epoch_size * 0.5:,.0f} draws (baseline: 75,736)")


if __name__ == "__main__":   # required on Windows
    main()
