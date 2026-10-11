"""
D0 integrity check. READ-ONLY: opens files, writes nothing, changes nothing.

Run after tools/bn_recalibrate.py, especially when the process died during shutdown, to prove the
outputs are complete and that the recalibrated copies differ from the source checkpoint in the
BatchNorm buffers ONLY.

  .venv\\Scripts\\python tools\\check_d0_integrity.py --dir runs\\exp001_bn_d0 --source runs\\experiment_001\\best.pth

Checks
  1  d0_results.json parses, lists every expected arm, and carries complete metrics
  2  each bn_*.pth loads, with no NaN/Inf anywhere and BN variances strictly positive
  3  every learned weight in each copy is bit-identical to the source checkpoint
  4  the BN buffers did change (a copy identical in buffers too would mean recalibration did nothing)
  5  the source checkpoint's SHA-256 still matches the value recorded during the run
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BUFFER_SUFFIX = ("running_mean", "running_var", "num_batches_tracked")
EXPECTED = ["hw-clean", "hw-aug", "synth", "mixed"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_state(sd):
    w = {k: v for k, v in sd.items() if not k.endswith(BUFFER_SUFFIX)}
    b = {k: v for k, v in sd.items() if k.endswith(BUFFER_SUFFIX)}
    return w, b


def tensors_equal(a, b):
    if set(a) != set(b):
        return False, "different keys"
    for k in sorted(a):
        x, y = a[k].detach().cpu(), b[k].detach().cpu()
        if x.shape != y.shape:
            return False, f"{k}: shape {tuple(x.shape)} vs {tuple(y.shape)}"
        if not torch.equal(x, y):
            return False, f"{k}: values differ"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/exp001_bn_d0")
    ap.add_argument("--source", default="runs/experiment_001/best.pth")
    ap.add_argument("--arms", nargs="+", default=EXPECTED)
    a = ap.parse_args()
    d, src = ROOT / a.dir, ROOT / a.source
    ok = True

    print("D0 integrity check (read-only)\n")
    # ---------------------------------------------------------------- 1: results file
    rj = d / "d0_results.json"
    res = None
    if not rj.exists():
        print(f"  FAIL  {rj} is missing"); ok = False
    else:
        try:
            res = json.loads(rj.read_text(encoding="utf-8"))
            arms = res.get("arms", {})
            missing = [x for x in ["as-is"] + a.arms if x not in arms]
            print(f"  {'OK  ' if not missing else 'FAIL'}  d0_results.json parses, {rj.stat().st_size:,} bytes, "
                  f"arms present: {', '.join(arms)}" + (f"   MISSING: {missing}" if missing else ""))
            ok &= not missing
            for name, arm in arms.items():
                cells = arm.get("eval", {})
                bad = [k for k, v in cells.items() if not all(f in v for f in ("n", "cer", "wer"))]
                line = "  ".join(f"{k} CER {100 * v['cer']:.4f}% WER {100 * v['wer']:.4f}% (n={v['n']})"
                                 for k, v in cells.items())
                print(f"        {name:<9} {line}" + (f"   INCOMPLETE: {bad}" if bad else ""))
                ok &= not bad and bool(cells)
        except json.JSONDecodeError as e:
            print(f"  FAIL  d0_results.json is not valid JSON (truncated write?): {e}"); ok = False

    # ---------------------------------------------------------------- 5: source checkpoint
    if not src.exists():
        print(f"\n  FAIL  source checkpoint {src} is missing"); sys.exit(1)
    src_hash = sha256(src)
    recorded = (res or {}).get("sha256")
    print(f"\n  source {src}  sha256 {src_hash}")
    if recorded:
        print(f"  {'OK  ' if recorded == src_hash else 'FAIL'}  matches the value recorded during the D0 run")
        ok &= recorded == src_hash
    else:
        print("  note  no recorded hash available to compare against")

    ck = torch.load(src, map_location="cpu", weights_only=False)
    src_w, src_b = split_state(ck["model"])
    print(f"  source checkpoint: epoch {ck.get('epoch')}, {len(src_w)} weight tensors, {len(src_b)} BN buffer tensors")

    # ---------------------------------------------------------------- 2-4: each copy
    for arm in a.arms:
        p = d / f"bn_{arm}.pth"
        print(f"\n  [{arm}] {p.name}")
        if not p.exists():
            print("        FAIL  file missing"); ok = False; continue
        try:
            c = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"        FAIL  will not load ({type(e).__name__}: {e}) - likely truncated"); ok = False; continue
        print(f"        OK    loads, {p.stat().st_size:,} bytes")
        w, b = split_state(c["model"])
        finite = all(torch.isfinite(v.float()).all() for v in c["model"].values())
        var_ok = all((v > 0).all() for k, v in b.items() if k.endswith("running_var"))
        print(f"        {'OK  ' if finite and var_ok else 'FAIL'}  all values finite: {finite}; "
              f"all BN variances > 0: {var_ok}")
        ok &= finite and var_ok
        same_w, why = tensors_equal(src_w, w)
        print(f"        {'OK  ' if same_w else 'FAIL'}  learned weights bit-identical to the source"
              + ("" if same_w else f" - {why}"))
        ok &= same_w
        same_b, _ = tensors_equal(src_b, b)
        print(f"        {'OK  ' if not same_b else 'FAIL'}  BN buffers differ from the source "
              f"(recalibration took effect): {not same_b}")
        ok &= not same_b
        meta = c.get("bn_recalibrated")
        print(f"        {'OK  ' if meta else 'note'}  recalibration metadata: {meta}")

    print(f"\n  RESULT: {'all checks passed' if ok else 'ONE OR MORE CHECKS FAILED'}")
    print("  (nothing was written or modified by this script)")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
