"""
Experiment: EXPERIMENTAL angle-aware line finding (exp_angle_lines) vs the default horizontal bands, on generated pages.

    <python> -B dev_tools/slant_experiments.py [--pages 3] [--out reports/diagnostic_implementation/experiments_slant_lines]

Conditions:
  - from tests/slant_fixtures.py (known line geometry): mixed angles up to +-1.5 and +-3 degrees (per page, from the
    seed), uniform skew 2 and 4 degrees, uneven spacing, curved lines, perspective, ruling under mixed-angle lines;
  - the 12 conditions of dev_tools/synthetic_conditions.py (checks that nothing else gets worse).
Splits: development seeds 2000+ and a fresh held-out split, seeds 9500+ (never used by any earlier run). Every page is
written once as PNG; both variants read IDENTICAL bytes through the app's decoder. Recognizer: registered Experiment
002-A checkpoint, production functions (wordocr.fullpage). Labels (text, word boxes) come from the generator.

Metrics per page and variant: segmentation vs labels (dev_tools/segmentation_experiments.run_variant: matched at IoU
>= 0.5, missed, split, merged, spurious, crop integrity), page CER (exact text) and CER without whitespace, and WER
(word-level edit distance over the page's words in reading order).

DECISION RULE (fixed here before the run, the same rule as the rule-removal experiment): the variant is a candidate if,
on DEVELOPMENT, it lowers page CER on at least one target condition (mixed_1p5, mixed_3) by >= 1 point without raising
any other condition's page CER by more than 0.5 point or lowering its matched rate by more than 1 point; it is
confirmed if the same holds on HELD-OUT. A confirmed candidate stays OFF by default: synthetic pages do not show
real-page accuracy.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import editdistance
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dev_tools"))
sys.path.insert(0, str(ROOT / "tests"))
import segmentation_experiments as se  # noqa: E402
import slant_fixtures as sf  # noqa: E402
import synthetic_conditions as sc  # noqa: E402
from wordocr import checkpoints, inputs, service  # noqa: E402
from wordocr.fullpage import PageSettings  # noqa: E402

VARIANTS = {"baseline": PageSettings(), "angle_lines": PageSettings(exp_angle_lines=True)}
TARGETS = ("mixed_1p5", "mixed_3")


def slant_condition(name: str, seed: int):
    rng = np.random.default_rng(seed + 17)
    if name == "mixed_1p5":
        return sf.make_slanted_page(seed=seed, angles=list(np.round(rng.uniform(-1.5, 1.5, 6), 2)))
    if name == "mixed_3":
        return sf.make_slanted_page(seed=seed, angles=list(np.round(rng.uniform(-3, 3, 6), 2)))
    if name == "skew_2":
        return sf.make_slanted_page(seed=seed, angles=[2.0] * 6)
    if name == "skew_4":
        return sf.make_slanted_page(seed=seed, angles=[-4.0] * 6)
    if name == "uneven_spacing":
        return sf.make_slanted_page(seed=seed, angles=[1.0] * 6, pitch=[int(v) for v in rng.integers(58, 125, 5)])
    if name == "curved_14":
        return sf.make_slanted_page(seed=seed, angles=[0.0] * 5, curve_amp=14)
    if name == "perspective_012":
        return sf.make_slanted_page(seed=seed, angles=[0.0] * 6, perspective=0.12)
    if name == "ruled_below_mixed":
        return sf.make_slanted_page(seed=seed, angles=list(np.round(rng.uniform(-2, 2, 6), 2)), rule="below")
    raise KeyError(name)


SLANT = ["mixed_1p5", "mixed_3", "skew_2", "skew_4", "uneven_spacing", "curved_14", "perspective_012", "ruled_below_mixed"]


def make(cond, seed):
    """(gray, words [(text, box)], text)"""
    if cond in SLANT:
        p = slant_condition(cond, seed)
        return p.gray, [(t, tuple(b)) for t, b in p.words], p.text
    p = sc.make(cond, seed)
    return p.gray, [(t, tuple(b)) for t, b in p.words], p.text


def wer(pred: str, ref: str):
    p, r = pred.split(), ref.split()
    return editdistance.eval(p, r), len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--out", default="reports/diagnostic_implementation/experiments_slant_lines")
    a = ap.parse_args()
    out = ROOT / a.out
    if (out / "results.json").exists():
        sys.exit(f"{out} already holds results; choose a new --out (evidence is never overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    model = service.load_model(checkpoints.DEFAULT_ID, "auto")
    conds = SLANT + list(sc.CONDITIONS)
    manifest, hashes = [], {}
    for split, base in (("development", 2000), ("heldout", 9500)):
        (out / "pages" / split).mkdir(parents=True, exist_ok=True)
        for ci, cond in enumerate(conds):
            for k in range(a.pages):
                seed = base + ci * 10 + k
                gray, words, text = make(cond, seed)
                data = cv2.imencode(".png", gray)[1].tobytes()
                h = hashlib.sha256(data).hexdigest()
                name = f"{cond}_{seed}.png"
                (out / "pages" / split / name).write_bytes(data)
                (out / "pages" / split / f"{cond}_{seed}.labels.json").write_text(
                    json.dumps({"text": text, "words": words, "condition": cond, "seed": seed, "label_source": "generator"}, ensure_ascii=False), encoding="utf-8")
                manifest.append({"split": split, "condition": cond, "seed": seed, "file": f"pages/{split}/{name}", "sha256": h, "words": len(words)})
                hashes.setdefault(h, []).append(f"{split}/{name}")
    dup = {h: v for h, v in hashes.items() if len(v) > 1}
    (out / "manifest.json").write_text(json.dumps({"pages": manifest, "duplicate_images": dup, "decision_rule": __doc__.split("DECISION RULE")[1].strip(),
                                                   "variants": {k: v.experimental() for k, v in VARIANTS.items()}}, indent=1), encoding="utf-8")
    results, t0 = [], time.time()
    for m in manifest:
        data = (out / m["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == m["sha256"]
        lab = json.loads((out / m["file"].replace(".png", ".labels.json")).read_text(encoding="utf-8"))
        page = sc.SyntheticPage(gray=None, words=[(t, tuple(b)) for t, b in lab["words"]], text=lab["text"], condition=m["condition"], seed=m["seed"])
        d = inputs.decode_upload(Path(m["file"]).name, data)
        for vname, vs in VARIANTS.items():
            row = {"split": m["split"], "condition": m["condition"], "seed": m["seed"], "variant": vname, "input_sha256": m["sha256"]}
            try:
                seg, res, _, met = se.run_variant(model, d, page, vname, vs)
                row.update(met)
                we, wn = wer(res.text, page.text)
                row.update(word_errors=we, ref_words=wn, checkpoint_sha256=res.model["checkpoint_sha256"],
                           lines_found=sum(len(ls) for ls in seg.lines),
                           tracked_blocks=sum(1 for ds in seg.word_diagnostics if ds and ds[0].get("line_source") == "tracked"))
            except Exception as e:                      # noqa: BLE001 - recorded per page, never hidden
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            results.append(row)
        print(f"{m['split']:11s} {m['condition']:22s} {m['seed']}  {time.time() - t0:6.0f}s", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    summarise(out, results, conds)


def agg(rows):
    s = lambda k: sum(r.get(k, 0) for r in rows)
    n = s("ref_chars")
    return {"pages": len(rows), "errors": sum("error" in r for r in rows), "truth_words": s("truth_words"), "matched": s("matched"),
            "missed": s("missed"), "split_words": s("split_words"), "merged": s("merged"), "spurious": s("spurious"),
            "matched_rate": round(s("matched") / max(1, s("truth_words")), 4), "page_CER": round(s("char_errors") / n, 4) if n else None,
            "page_CER_no_whitespace": round(s("char_errors_no_ws") / max(1, s("ref_chars_no_ws")), 4),
            "WER": round(s("word_errors") / max(1, s("ref_words")), 4), "tracked_pages": sum(1 for r in rows if r.get("tracked_blocks"))}


def summarise(out, results, conds):
    table = [{"split": sp, "condition": c, "variant": v, **agg([r for r in results if r["split"] == sp and r["condition"] == c and r["variant"] == v])}
             for sp in ("development", "heldout") for c in conds for v in VARIANTS]
    with open(out / "results_by_condition.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    T = {(r["split"], r["condition"], r["variant"]): r for r in table}
    res = {}
    for sp in ("development", "heldout"):
        gains, reg = [], []
        for c in conds:
            b, x = T[(sp, c, "baseline")], T[(sp, c, "angle_lines")]
            if b["page_CER"] is None or x["page_CER"] is None:
                continue
            dc, dm = x["page_CER"] - b["page_CER"], x["matched_rate"] - b["matched_rate"]
            if c in TARGETS and dc <= -0.01:
                gains.append([c, round(dc, 4)])
            if dc > 0.005 or dm < -0.01:
                reg.append([c, round(dc, 4), round(dm, 4)])
        res[sp] = {"gains_on_targets": gains, "regressions": reg, "passes": bool(gains) and not reg}
    res["verdict"] = ("CANDIDATE (confirmed on held-out; default not changed)" if res["development"]["passes"] and res["heldout"]["passes"]
                      else "REJECTED" if not res["development"]["passes"] else "NOT CONFIRMED on held-out")
    (out / "decision.json").write_text(json.dumps({"angle_lines": res}, indent=1), encoding="utf-8")
    for r in table:
        print(f"{r['split']:11s} {r['condition']:22s} {r['variant']:12s} match {r['matched_rate']:.3f} split {r['split_words']:3d} merged {r['merged']:3d} "
              f"missed {r['missed']:3d} CER {r['page_CER']} WER {r['WER']} tracked_pages {r['tracked_pages']}")
    print(json.dumps({"angle_lines": res}, indent=1))


if __name__ == "__main__":
    main()
