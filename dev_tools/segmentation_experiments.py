"""
Phase C-E experiment: segmentation variants on synthetic pages under controlled conditions, scored against the
generator's own labels, with the production recognizer and the official checkpoint.

    <python> -B dev_tools/segmentation_experiments.py [--pages 3] [--out reports/diagnostic_implementation/experiments]

Splits (fixed seeds, never tuned on held-out):  development seeds 1000+  /  held-out seeds 5000+.
Every page is written once as PNG into <out>/pages/<split>/ with its labels; both splits are hashed and checked for
identical images (leakage). Variants are run on IDENTICAL bytes (the PNG is decoded by the app's own decoder each time).

Per page and variant:
  segmentation vs labels   matched (IoU >= 0.5, one-to-one), missed, split, merged, spurious, mean IoU
  crop integrity           wordocr.crop_compare.integrity of every labelled word: intact / clipped / split / merged / missed
  recognition              page CER on the exact page text (primary; Unicode, spaces and line breaks kept);
                           CER after collapsing all whitespace (secondary); words correct among matched regions
  A vs B (baseline only)   condition A = the labelled box cut from the page; condition B = the pipeline's crop for it;
                           both read by the same recognizer: tells segmentation failures from recognition failures
Nothing here changes a default. Writes results.json, results_by_condition.csv, decision.json and overlays.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import editdistance
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dev_tools"))
import synthetic_conditions as sc  # noqa: E402
from wordocr import checkpoints, crop_compare, fullpage, inputs, service  # noqa: E402

VARIANTS = {
    "baseline": fullpage.PageSettings(),
    "rule_removal": fullpage.PageSettings(exp_rule_removal=True),
    "rule_removal+crop_fill": fullpage.PageSettings(exp_rule_removal=True, exp_crop_rule_fill=True),
    "contrast_clahe": fullpage.PageSettings(exp_contrast_clahe=True),
    "wider_margin_0.4h": "margin:0.4",
}
TARGETS = {"rule_removal": ("ruled_below", "ruled_through"), "rule_removal+crop_fill": ("ruled_below", "ruled_through"),
           "contrast_clahe": ("low_contrast", "shadow", "blur"), "wider_margin_0.4h": ()}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def cover(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def seg_metrics(truth, pred):
    pairs = sorted(((iou(t[1], p), i, j) for i, t in enumerate(truth) for j, p in enumerate(pred)), reverse=True)
    ti, pj, matched = set(), set(), []
    for v, i, j in pairs:
        if v < 0.5:
            break
        if i not in ti and j not in pj:
            ti.add(i); pj.add(j); matched.append((i, j, v))
    split = sum(1 for t in truth if sum(cover(t[1], p) > 0.15 for p in pred) >= 2)
    merged = sum(1 for p in pred if sum(cover(t[1], p) > 0.5 for t in truth) >= 2)
    missed = sum(1 for t in truth if not any(cover(t[1], p) > 0.15 for p in pred))
    spurious = sum(1 for p in pred if not any(cover(p, t[1]) > 0.15 for t in truth))
    return {"truth_words": len(truth), "pred_regions": len(pred), "matched": len(matched), "missed": missed, "split_words": split, "merged": merged,
            "spurious": spurious, "mean_iou_matched": round(float(np.mean([v for _, _, v in matched])), 4) if matched else 0.0}, matched


def cer(pred, ref):
    return editdistance.eval(pred, ref), len(ref)


def run_variant(model, d, page, name, settings):
    if isinstance(settings, str):                       # margin variant: same segmentation, margin 0.4 x text height
        seg = fullpage.segment_page(d, fullpage.PageSettings())
        settings = fullpage.PageSettings(margin=int(np.clip(round(0.4 * seg.h), 2, 24)))
    else:
        seg = fullpage.segment_page(d, settings)
    res = fullpage.recognize_page(model, d, seg, settings)
    pred = [w.box for w in res.words]
    m, matched = seg_metrics(page.words, pred)
    e, n = cer(res.text, page.text)
    e2, n2 = cer("".join(res.text.split()), "".join(page.text.split()))
    words_ok = sum(res.words[j].text == page.words[i][0] for i, j, _ in matched)
    regions = [{"id": w.id, "box_original": w.box} for w in res.words]
    integ = {}
    for t in page.words:
        v = crop_compare.integrity(t[1], regions)["verdict"].split(":")[0]
        integ[v] = integ.get(v, 0) + 1
    return seg, res, settings, {**m, "crop_integrity": integ, "char_errors": e, "ref_chars": n, "char_errors_no_ws": e2, "ref_chars_no_ws": n2,
                                "matched_words_correct": words_ok, "flags": list(res.segmentation["flags"])}


def a_vs_b(model, d, seg, res, page):
    """Condition A: labelled box cut from the page; B: the pipeline crop of the best-overlapping region. Same recognizer."""
    samples = [crop_compare.Sample(id=f"w{i:03d}", box=t[1], reference=t[0], reference_confirmed=True, reference_source="generator")
               for i, t in enumerate(page.words)]
    rep = crop_compare.compare(model, d, seg, res.margin_px, samples, result=res)
    s = rep["summary"]
    return {k: s.get(k) for k in ("samples", "A_manual_words_correct", "B_pipeline_words_correct", "A_manual_CER", "B_pipeline_CER", "reference_characters", "crop_integrity", "pattern")} | \
           {"consistency_with_page_result": all(r.get("pipeline_prediction_matches_page_result", True) for r in rep["samples"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--out", default="reports/diagnostic_implementation/experiments")
    ap.add_argument("--heldout-base", type=int, default=5000, help="seed base of the held-out split (a NEW base gives an unseen split)")
    a = ap.parse_args()
    out = ROOT / a.out
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    model = service.load_model(checkpoints.DEFAULT_ID, "auto")
    splits = {"development": 1000, "heldout": a.heldout_base}
    manifest, hashes = [], {}
    for split, base in splits.items():
        (out / "pages" / split).mkdir(parents=True, exist_ok=True)
        for ci, cond in enumerate(sc.CONDITIONS):
            for k in range(a.pages):
                seed = base + ci * 10 + k
                p = sc.make(cond, seed)
                data = cv2.imencode(".png", p.gray)[1].tobytes()
                h = hashlib.sha256(data).hexdigest()
                name = f"{cond}_{seed}.png"
                (out / "pages" / split / name).write_bytes(data)
                (out / "pages" / split / f"{cond}_{seed}.labels.json").write_text(json.dumps({"text": p.text, "words": p.words, "condition": cond, "seed": seed,
                                                                                               "label_source": "generator (typeset text and boxes)"}, ensure_ascii=False), encoding="utf-8")
                manifest.append({"split": split, "condition": cond, "seed": seed, "file": f"pages/{split}/{name}", "sha256": h, "words": len(p.words)})
                hashes.setdefault(h, []).append(f"{split}/{name}")
    dup = {h: v for h, v in hashes.items() if len(v) > 1}
    cross = {h: v for h, v in dup.items() if len({x.split("/")[0] for x in v}) > 1}
    word_overlap = None
    (out / "manifest.json").write_text(json.dumps({"pages": manifest, "duplicate_images": dup, "duplicates_across_splits": cross,
                                                   "leakage_note": "splits differ by seed; both draw words from the same fixed list of common Tamil words "
                                                                   "(tests/fullpage_fixtures.WORDS) and use fonts the model was trained with: the recognizer "
                                                                   "part of these measurements is NOT independent of training"}, indent=1), encoding="utf-8")
    results = []
    t0 = time.time()
    for m in manifest:
        data = (out / m["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == m["sha256"]
        lab = json.loads((out / m["file"].replace(".png", ".labels.json")).read_text(encoding="utf-8"))
        page = sc.SyntheticPage(gray=None, words=[(t, tuple(b)) for t, b in lab["words"]], text=lab["text"], condition=m["condition"], seed=m["seed"])
        d = inputs.decode_upload(Path(m["file"]).name, data)
        for vname, vs in VARIANTS.items():
            row = {"split": m["split"], "condition": m["condition"], "seed": m["seed"], "variant": vname, "input_sha256": m["sha256"]}
            try:
                seg, res, used, met = run_variant(model, d, page, vname, vs)
                row.update(met)
                row["checkpoint_sha256"] = res.model["checkpoint_sha256"]
                if vname == "baseline":
                    row["a_vs_b"] = a_vs_b(model, d, seg, res, page)
                if m["seed"] % 10 == 0 and (vname == "baseline" or m["condition"] in TARGETS.get(vname, ())):
                    img = fullpage.overlay(seg, res.words)
                    cv2.imwrite(str(out / "overlays" / f"{m['split']}_{m['condition']}_{m['seed']}_{vname}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            except fullpage.PageError as e:
                row.update({"error": str(e)[:200]})
            results.append(row)
        print(f"{m['split']:11s} {m['condition']:22s} {m['seed']}  {time.time() - t0:6.0f}s", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    summarise(out, results)


def agg(rows):
    s = lambda k: sum(r.get(k, 0) for r in rows)
    n = s("ref_chars")
    return {"pages": len(rows), "errors": sum("error" in r for r in rows), "truth_words": s("truth_words"), "matched": s("matched"), "missed": s("missed"),
            "split_words": s("split_words"), "merged": s("merged"), "spurious": s("spurious"),
            "matched_rate": round(s("matched") / max(1, s("truth_words")), 4),
            "page_CER": round(s("char_errors") / n, 4) if n else None,
            "page_CER_no_whitespace": round(s("char_errors_no_ws") / max(1, s("ref_chars_no_ws")), 4),
            "matched_words_correct_rate": round(s("matched_words_correct") / max(1, s("matched")), 4)}


def summarise(out, results):
    table = []
    for split in ("development", "heldout"):
        for cond in sc.CONDITIONS:
            for v in VARIANTS:
                rows = [r for r in results if r["split"] == split and r["condition"] == cond and r["variant"] == v]
                table.append({"split": split, "condition": cond, "variant": v, **agg(rows)})
    with open(out / "results_by_condition.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    ab = {}
    for split in ("development", "heldout"):
        for cond in sc.CONDITIONS:
            rows = [r["a_vs_b"] for r in results if r["split"] == split and r["condition"] == cond and r["variant"] == "baseline" and "a_vs_b" in r]
            if rows:
                n = sum(r["samples"] for r in rows)
                ch = sum(r["reference_characters"] for r in rows)
                ab[f"{split}/{cond}"] = {"words": n, "A_manual_correct": sum(r["A_manual_words_correct"] for r in rows),
                                         "B_pipeline_correct": sum(r["B_pipeline_words_correct"] for r in rows),
                                         "A_CER": round(sum(r["A_manual_CER"] * r["reference_characters"] for r in rows) / ch, 4),
                                         "B_CER": round(sum(r["B_pipeline_CER"] * r["reference_characters"] for r in rows) / ch, 4),
                                         "pipeline_reread_consistent": all(r["consistency_with_page_result"] for r in rows)}
    # decision rule, fixed before looking at held-out: a variant is a candidate if, on DEVELOPMENT, it lowers page CER
    # on at least one target condition by >= 1 point (absolute) without raising any other condition's page CER by more
    # than 0.5 point or lowering its matched rate by more than 1 point; it is confirmed if the same holds on HELD-OUT.
    T = {(r["split"], r["condition"], r["variant"]): r for r in table}
    decision = {}
    for v in VARIANTS:
        if v == "baseline":
            continue
        res = {}
        for split in ("development", "heldout"):
            gains, regress = [], []
            for cond in sc.CONDITIONS:
                b, x = T[(split, cond, "baseline")], T[(split, cond, v)]
                if b["page_CER"] is None or x["page_CER"] is None:
                    continue
                d_cer = x["page_CER"] - b["page_CER"]
                d_match = x["matched_rate"] - b["matched_rate"]
                if cond in TARGETS[v] and d_cer <= -0.01:
                    gains.append((cond, round(d_cer, 4)))
                if d_cer > 0.005 or d_match < -0.01:
                    regress.append((cond, round(d_cer, 4), round(d_match, 4)))
            res[split] = {"gains_on_targets": gains, "regressions": regress, "passes": bool(gains) and not regress}
        res["verdict"] = ("CANDIDATE (confirmed on held-out; default not changed)" if res["development"]["passes"] and res["heldout"]["passes"]
                          else "REJECTED" if not res["development"]["passes"] else "NOT CONFIRMED on held-out")
        decision[v] = res
    (out / "a_vs_b.json").write_text(json.dumps(ab, indent=1), encoding="utf-8")
    (out / "decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")
    for r in table:
        if r["variant"] == "baseline" or r["condition"] in TARGETS.get(r["variant"], ()):
            print(f"{r['split']:11s} {r['condition']:22s} {r['variant']:22s} match {r['matched_rate']:.3f} split {r["split_words"]:3d} merged {r['merged']:3d} "
                  f"missed {r['missed']:3d} CER {r['page_CER']} CERnows {r['page_CER_no_whitespace']}")
    print(json.dumps(ab, indent=1))
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
