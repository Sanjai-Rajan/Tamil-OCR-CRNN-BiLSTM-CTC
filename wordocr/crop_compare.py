"""
Controlled comparison: a carefully placed MANUAL word crop (condition A) against the crop the full-page pipeline cut
for the same word (condition B), both read by the same recognizer in the same way.

    samples = [Sample(id="S01", box=(x1, y1, x2, y2), reference="...", reference_confirmed=True), ...]   # ORIGINAL page pixels
    report  = compare(model, decoded, seg, result_margin_px, samples)

Same checkpoint, same preprocessing, same decoding: each crop goes to model.engine.recognize_batch([crop], batch_size=1),
exactly as wordocr.fullpage.recognize_page does. Condition A's crop is cut from the uploaded page's own pixels (the
original frame) with the same margin rule as the pipeline; condition B is the pipeline crop of the region that overlaps
the manual box most, cut by pageocr.pipeline.ocr_crops.

Kept apart on purpose, in separate fields:
    reference / reference_confirmed   entered and confirmed by a person (or, for generated pages, the generator's own text)
    prediction_manual / prediction_pipeline   what the recognizer produced
    crop integrity                    how the pipeline's region relates to the manual box (overlap, split, merge)
A reference that is not confirmed is never scored. A prediction is never used as a reference.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from pageocr.engine import InvalidCropError, validate_crop


@dataclass
class Sample:
    id: str
    box: tuple                      # (x1, y1, x2, y2) exclusive, ORIGINAL page pixels, drawn by a person tightly around one word
    reference: str = ""             # the person's transcription; "" = not given
    reference_confirmed: bool = False
    reference_source: str = "human" # "human" or "generator" (synthetic page: the text the generator typeset)
    note: str = ""


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _cover(a, b):
    """Share of box a covered by box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def edit_distance(a: str, b: str) -> int:
    import editdistance
    return int(editdistance.eval(a, b))


def _read(model, crop):
    try:
        validate_crop(crop)
    except InvalidCropError as e:
        return None, f"crop not readable: {e}"
    with model.lock, torch.inference_mode():
        rec = model.engine.recognize_batch([crop], batch_size=1)[0]
    return rec, ""


def integrity(sample_box, regions) -> dict:
    """How the pipeline's word regions relate to one manual word box (both in ORIGINAL coordinates)."""
    over = [(r, _cover(sample_box, r["box_original"]), _cover(r["box_original"], sample_box)) for r in regions]
    touching = [(r, a, b) for r, a, b in over if a > 0.05]
    best = max(touching, key=lambda t: _iou(sample_box, t[0]["box_original"]), default=None)
    out = {"regions_overlapping": [t[0]["id"] for t in touching], "best_region": best[0]["id"] if best else None,
           "best_iou": round(_iou(sample_box, best[0]["box_original"]), 3) if best else 0.0,
           "manual_box_covered_by_best": round(best[1], 3) if best else 0.0,
           "best_region_inside_manual_box": round(best[2], 3) if best else 0.0}
    if not touching:
        verdict = "missed: no pipeline region overlaps the word"
    elif len([t for t in touching if t[1] > 0.15]) >= 2:
        verdict = "split: the word is spread over several pipeline regions"
    elif best[2] < 0.6:
        verdict = "merged: the pipeline region is much larger than the word (other ink included)"
    elif best[1] < 0.9:
        verdict = "clipped: the pipeline region leaves part of the word out"
    else:
        verdict = "intact"
    out["verdict"] = verdict
    return out


def compare(model, decoded, seg, margin_px: int, samples, result=None) -> dict:
    """Read every sample twice (manual crop, pipeline crop) and describe both. `result` (PageResult) is optional; if given,
    its stored pipeline prediction must equal the re-read one (a consistency check, not a second opinion)."""
    from pageocr import pipeline
    regions = [{"index": i, "id": w.id, "box_original": tuple(int(v) for v in w.box_original)} for i, w in enumerate(seg.words)]
    pipe_crops = pipeline.ocr_crops(seg.ocr, seg.words, margin_px)
    H, W = decoded.gray.shape
    rows = []
    for s in samples:
        x1, y1, x2, y2 = (int(v) for v in s.box)
        if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
            rows.append({"id": s.id, "box": [x1, y1, x2, y2], "error": "the box is outside the page or empty"})
            continue
        crop_a = decoded.gray[max(0, y1 - margin_px):min(H, y2 + margin_px), max(0, x1 - margin_px):min(W, x2 + margin_px)].copy()
        rec_a, err_a = _read(model, crop_a)
        integ = integrity((x1, y1, x2, y2), regions)
        row = {"id": s.id, "box": [x1, y1, x2, y2], "note": s.note,
               "reference": s.reference, "reference_confirmed": bool(s.reference_confirmed and s.reference), "reference_source": s.reference_source,
               "prediction_manual": rec_a.text if rec_a else "", "error_manual": err_a, "crop_manual_size": list(crop_a.shape[::-1]),
               "integrity": integ, "pipeline_region": integ["best_region"]}
        if integ["best_region"] is not None:
            j = next(r["index"] for r in regions if r["id"] == integ["best_region"])
            rec_b, err_b = _read(model, pipe_crops[j])
            row.update(prediction_pipeline=rec_b.text if rec_b else "", error_pipeline=err_b, crop_pipeline_size=list(pipe_crops[j].shape[::-1]))
            if result is not None:
                row["pipeline_prediction_matches_page_result"] = result.words[j].text == row["prediction_pipeline"]
        else:
            row.update(prediction_pipeline=None, error_pipeline="no pipeline region", crop_pipeline_size=None)
        if row["reference_confirmed"]:
            ref = s.reference
            row["manual_correct"] = row["prediction_manual"] == ref
            row["pipeline_correct"] = row["prediction_pipeline"] == ref
            row["manual_char_errors"] = edit_distance(row["prediction_manual"], ref)
            row["pipeline_char_errors"] = edit_distance(row["prediction_pipeline"] or "", ref)
            row["reference_chars"] = len(ref)
        rows.append(row)
    return {"samples": rows, "summary": summarise(rows), "margin_px": margin_px, "model": model.info()}


def summarise(rows) -> dict:
    scored = [r for r in rows if r.get("reference_confirmed")]
    verdicts = {}
    for r in rows:
        if "integrity" in r:
            v = r["integrity"]["verdict"].split(":")[0]
            verdicts[v] = verdicts.get(v, 0) + 1
    out = {"samples": len(rows), "with_confirmed_reference": len(scored), "crop_integrity": verdicts,
           "predictions_identical_A_vs_B": sum(1 for r in rows if "prediction_manual" in r and r.get("prediction_pipeline") == r["prediction_manual"])}
    if scored:
        n = sum(r["reference_chars"] for r in scored)
        out.update({"A_manual_words_correct": sum(r["manual_correct"] for r in scored), "B_pipeline_words_correct": sum(r["pipeline_correct"] for r in scored),
                    "A_manual_CER": round(sum(r["manual_char_errors"] for r in scored) / n, 4) if n else None,
                    "B_pipeline_CER": round(sum(r["pipeline_char_errors"] for r in scored) / n, 4) if n else None,
                    "reference_characters": n,
                    "pattern": pattern(scored)})
    else:
        out["note"] = "no confirmed reference: nothing is scored; only crop integrity and A/B agreement are reported"
    return out


def pattern(scored) -> str:
    """The decision rule of the work order, applied to the scored samples (a description, not a verdict on the system)."""
    a = sum(r["manual_correct"] for r in scored) / len(scored)
    b = sum(r["pipeline_correct"] for r in scored) / len(scored)
    if a >= 0.8 and b < a - 0.15:
        return "manual crops mostly correct, pipeline crops worse: segmentation / cropping is the main problem"
    if a < 0.6:
        return "manual crops also often wrong: recognition (or image quality) is a limiting factor"
    if abs(a - b) <= 0.15:
        return "manual and pipeline crops perform alike: cropping is not the main difference"
    return "mixed: inspect per sample"
