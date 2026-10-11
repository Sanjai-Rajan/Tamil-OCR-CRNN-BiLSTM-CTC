"""
Diagnostics of one full-page OCR run: what was found, what was cut, what was read, and what looks suspicious.

    diag   = analyse(decoded, seg, result, settings)      # plain data; no pixels changed, nothing recognised again
    table  = region_rows(diag)                            # one row per word region, for a sortable table
    img    = overlay_original(decoded, diag) / overlay_ocr(seg, diag)
    data   = bundle_zip(decoded, seg, result, settings, diag, include_input=True)   # a reproducible evidence bundle

Everything here is READ-ONLY with respect to the OCR run: it takes the Segmentation (pageocr.pipeline) and the
PageResult (wordocr.fullpage) the normal path produced and describes them. The crops it exports are cut by the same
function the recognizer's crops are cut by (pageocr.pipeline.ocr_crops, same margin), and their SHA-256 is recorded.
Nothing is fabricated: a value the pipeline does not produce is reported as None ("unavailable").

WARNINGS ARE HEURISTICS. Each says "look at this region", not "this region is wrong". Their rules are in WARNING_RULES
and in the code below; the thresholds are fixed numbers, recorded in every export.
"""
import csv
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

from pageocr import ROOT, line_geometry, pipeline
from pageocr.engine import natural_width
from src import preprocess as pp
from wordocr import VERSION

DIAG_VERSION = "wordocr-diagnostics-1.1.0"     # 1.1.0: angle-aware line geometry (pageocr.line_geometry) in the record, overlays and bundle
CODE_FILES = ("app.py", "wordocr", "pageocr", "src/model", "src/preprocess.py", "src/charset.py")
WARNING_RULES = {
    "not_read": "the region was not recognised (crop not readable or recognition failed)",
    "empty_prediction": "the recognizer emitted no character for this crop",
    "clipped_at_page_edge": "the crop margin was cut short by the image border on at least one side",
    "ink_at_crop_edge": "dark pixels touch a crop border over at least 15 % of that border: a mark may be cut off, or a neighbour reaches in",
    "overlaps_neighbour": "the padded crop overlaps the ink box of another word region: the crop may contain part of it",
    "taller_than_line": "the box is more than 1.6 x the median word height of its line: possibly two lines merged, or a large mark",
    "very_wide": "width / height is more than 3 x the page's median word aspect: possibly several words merged",
    "narrow_fragment": "a 'word' narrower than 0.45 x the text height: possibly a fragment of a word",
    "line_cut_guessed": "its line was cut out of a band of touching lines; the cut is a guess",
    "rule_line_in_crop": "a crop row is dark across at least 85 % of the crop width (crop wider than 2 x text height): notebook ruling or underline suspected",
    "squeezed_to_max_width": f"the crop is wider than the model's maximum input ({pp.MAX_W} px at {pp.TARGET_H} px height) and is squeezed",
}


def code_fingerprint() -> dict:
    """SHA-256 over the sorted source files that take part in a run (there is no version-control revision here)."""
    files = []
    for item in CODE_FILES:
        p = ROOT / item
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + hashlib.sha256(f.read_bytes()).digest())
    return {"sha256": h.hexdigest(), "files": len(files), "wordocr_version": VERSION, "pipeline_version": pipeline.PIPELINE_VERSION,
            "diagnostics_version": DIAG_VERSION}


def _dark_mask(crop: np.ndarray) -> np.ndarray:
    """Pixels clearly darker than the crop's paper: below the midpoint of the paper level and the darkest ink."""
    paper = float(np.percentile(crop, 90))
    ink = float(np.percentile(crop, 2))
    if paper - ink < 25:                                   # no clear ink in this crop
        return np.zeros(crop.shape, bool)
    return crop < (paper + ink) / 2.0


def _crop_box(box, pad, W, H):
    x1, y1, x2, y2 = box
    return (max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad))


def analyse(decoded, seg, result, settings) -> dict:
    """The diagnostic record of one run. `result` is the PageResult of wordocr.fullpage.recognize_page for `seg`."""
    if len(result.words) != len(seg.words):
        raise ValueError("the result does not belong to this segmentation (different number of regions)")
    H, W = seg.ocr.gray.shape
    pad = int(result.margin_px)
    crops = pipeline.ocr_crops(seg.ocr, seg.words, pad)           # exactly the recognizer's crops
    hts = np.array([w.box[3] - w.box[1] for w in seg.words], float) if seg.words else np.zeros(0)
    asp = np.array([(w.box[2] - w.box[0]) / max(1, w.box[3] - w.box[1]) for w in seg.words if w.kind == "word"], float)
    med_aspect = float(np.median(asp)) if len(asp) else None
    line_med = {}
    for w, hgt in zip(seg.words, hts):
        line_med.setdefault((w.block_index, w.line_index), []).append(hgt)
    line_med = {k: float(np.median(v)) for k, v in line_med.items()}
    boxes = [w.box for w in seg.words]

    lines = []
    for bi, (b, ls) in enumerate(zip(seg.blocks, seg.lines)):
        for li, ln in enumerate(ls):
            d = seg.word_diagnostics[bi][li]
            lines.append({"block": bi, "line": li, "id": f"b{bi:02d}_l{li:03d}", "y1_ocr": int(ln.y1), "y2_ocr": int(ln.y2), "x1_ocr": int(b.x1), "x2_ocr": int(b.x2),
                          "height_px": int(ln.y2 - ln.y1), "forced_split": bool(ln.forced_split), "word_gap_threshold_px": d.get("threshold_px"),
                          "threshold_source": d.get("threshold_source"), "gaps_px": d.get("gaps", []),
                          "regions": [w.index for w in result.words if w.block == bi and w.line == li]})

    regions = []
    for i, (sw, pw, crop) in enumerate(zip(seg.words, result.words, crops)):
        assert sw.id == pw.id and tuple(sw.box) == tuple(pw.box_ocr), "region identity mismatch"
        cb = _crop_box(sw.box, pad, W, H)
        warns = []
        if not pw.ok:
            warns.append("not_read")
        elif pw.text == "":
            warns.append("empty_prediction")
        x1, y1, x2, y2 = sw.box
        if (x1 - pad < 0 or y1 - pad < 0 or x2 + pad > W or y2 + pad > H) and pad > 0:
            warns.append("clipped_at_page_edge")
        dark = _dark_mask(crop) if crop.size else np.zeros((0, 0), bool)
        if dark.size and min(dark.shape) >= 3:
            edges = (dark[0, :].mean(), dark[-1, :].mean(), dark[:, 0].mean(), dark[:, -1].mean())
            if max(edges) >= 0.15:
                warns.append("ink_at_crop_edge")
            if crop.shape[1] > 2 * seg.h and (dark.mean(axis=1) >= 0.85).any():
                warns.append("rule_line_in_crop")
        for j, ob in enumerate(boxes):
            if j != i and cb[0] < ob[2] and ob[0] < cb[2] and cb[1] < ob[3] and ob[1] < cb[3]:
                warns.append("overlaps_neighbour")
                break
        lm = line_med.get((sw.block_index, sw.line_index))
        if lm and (y2 - y1) > 1.6 * lm:
            warns.append("taller_than_line")
        if med_aspect and sw.kind == "word" and (x2 - x1) / max(1, y2 - y1) > 3 * med_aspect:
            warns.append("very_wide")
        if sw.kind == "word" and seg.h and (x2 - x1) < 0.45 * seg.h:
            warns.append("narrow_fragment")
        if sw.line_forced_split:
            warns.append("line_cut_guessed")
        if crop.size and natural_width(crop.shape[0], crop.shape[1]) > pp.MAX_W:
            warns.append("squeezed_to_max_width")
        regions.append({"index": i, "number": i + 1, "id": pw.id, "block": pw.block, "line": pw.line, "word": pw.word, "kind": pw.kind,
                        "box_original": list(pw.box), "box_ocr": list(pw.box_ocr), "crop_box_ocr": list(cb),
                        "box_width": int(x2 - x1), "box_height": int(y2 - y1), "crop_width": int(crop.shape[1]) if crop.size else 0,
                        "crop_height": int(crop.shape[0]) if crop.size else 0, "crop_sha256": hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest(),
                        "gap_before_px": int(sw.gap_before), "status": pw.status, "text": pw.text, "error": pw.error,
                        "uncalibrated_score": pw.uncalibrated_score, "warnings": warns})
    counts = {k: sum(k in r["warnings"] for r in regions) for k in WARNING_RULES}
    try:                                       # angle-aware line description; read-only, and never fatal to the record
        geo = line_geometry.line_geometry(seg)
    except Exception as e:                     # noqa: BLE001 - recorded, not hidden
        geo = {"version": line_geometry.GEOMETRY_VERSION, "error": f"{type(e).__name__}: {e}"}
    return {"diagnostics_version": DIAG_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "input": {"name": decoded.name, "sha256": decoded.sha256, "bytes": decoded.size_bytes, "format": decoded.format,
                      "width": decoded.width, "height": decoded.height},
            "frames": {"original": "pixels of the uploaded image (after EXIF rotation)",
                       "ocr": "the image the crops are cut from: the original, inverted if light-on-dark, rotated if deskewed"
                              + (f" (rotated by {seg.ocr.deskew_deg:+.2f} degrees)" if seg.ocr.deskew_deg else " (not rotated)")},
            "model": dict(result.model), "settings": dict(result.settings), "margin_px": pad, "segmentation": dict(result.segmentation),
            "code": code_fingerprint(), "warning_rules": WARNING_RULES, "warning_counts": counts,
            "regions_with_any_warning": sum(bool(r["warnings"]) for r in regions),
            "counts": result.counts, "lines": lines, "line_geometry": geo, "regions": regions, "page_text": result.text,
            "unavailable": {"per_character_alignment": "the recognizer does not report character positions",
                            "calibrated_confidence": "the score is the mean winning-class probability, not a probability of being correct",
                            "ground_truth": "no reference text is known for an uploaded page"}}


REGION_COLUMNS = ["number", "id", "block", "line", "word", "kind", "status", "text", "warnings", "box_original", "box_ocr", "crop_box_ocr",
                  "box_width", "box_height", "crop_width", "crop_height", "gap_before_px", "uncalibrated_score", "error", "crop_sha256"]


def region_rows(diag: dict) -> list:
    """Flat rows for a table or CSV (lists joined)."""
    out = []
    for r in diag["regions"]:
        row = {k: r[k] for k in REGION_COLUMNS}
        row["warnings"] = ", ".join(r["warnings"])
        for k in ("box_original", "box_ocr", "crop_box_ocr"):
            row[k] = " ".join(str(v) for v in r[k])
        out.append(row)
    return out


def _draw(img_gray, boxes_with_meta, h):
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    scale = max(0.35, min(1.2, h / 30.0)) if h else 0.5
    thick = max(1, int(round(h / 25.0))) if h else 1
    for (x1, y1, x2, y2), number, warned in boxes_with_meta:
        colour = (230, 120, 0) if warned else (200, 0, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2) - 1, int(y2) - 1), colour, thick)
        cv2.putText(img, str(number), (int(x1), max(int(10 * scale) + 2, int(y1) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, colour,
                    max(1, thick - 1), cv2.LINE_AA)
    return img


def overlay_ocr(seg, diag: dict, show_lines: bool = True, line_mode: str = "horizontal") -> np.ndarray:
    """The OCR-frame image with lines and word regions numbered as in the table (red; orange = has a warning).
    line_mode "horizontal": the segmentation's line bands (green), as before; "angle": the angle-aware line boxes of
    pageocr.line_geometry (green / blue alternating, magenta = no reliable local angle)."""
    img = _draw(seg.ocr.gray, [(r["box_ocr"], r["number"], bool(r["warnings"])) for r in diag["regions"]], seg.h)
    if show_lines and line_mode == "angle":
        return line_geometry.draw_lines(img, _geometry(diag), "ocr", "angle", seg)
    if show_lines:
        for ln in diag["lines"]:
            cv2.rectangle(img, (ln["x1_ocr"], ln["y1_ocr"]), (ln["x2_ocr"] - 1, ln["y2_ocr"] - 1), (0, 150, 0), 1)
    return img


def overlay_original(decoded, diag: dict, h: float, line_mode: str | None = None, seg=None) -> np.ndarray:
    """The uploaded page with the word regions in ORIGINAL coordinates, numbered as in the table. line_mode None: no
    lines (as before); "angle": the angle-aware line boxes; "horizontal": the segmentation's bands mapped into the
    original frame (a rotated quadrilateral where the page was deskewed). Needs seg for "horizontal"."""
    img = _draw(decoded.gray, [(r["box_original"], r["number"], bool(r["warnings"])) for r in diag["regions"]], h)
    if line_mode is None:
        return img
    return line_geometry.draw_lines(img, _geometry(diag), "original", line_mode, seg)


def _geometry(diag: dict) -> dict:
    g = diag.get("line_geometry")
    if not g or "error" in g:
        raise ValueError("no line geometry in this diagnostic record" + (f": {g['error']}" if g else ""))
    return g


def crop_context(seg, diag: dict, number: int, context_h: float = 1.5) -> np.ndarray:
    """The region's crop box (blue) and tight box (red) inside a wider window of the OCR image, for inspection."""
    r = diag["regions"][number - 1]
    H, W = seg.ocr.gray.shape
    m = int(round(context_h * (seg.h or 20)))
    x1, y1, x2, y2 = r["crop_box_ocr"]
    X1, Y1, X2, Y2 = max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)
    img = cv2.cvtColor(seg.ocr.gray[Y1:Y2, X1:X2], cv2.COLOR_GRAY2RGB)
    cv2.rectangle(img, (x1 - X1, y1 - Y1), (x2 - X1 - 1, y2 - Y1 - 1), (0, 90, 220), 1)
    bx1, by1, bx2, by2 = r["box_ocr"]
    cv2.rectangle(img, (bx1 - X1, by1 - Y1), (bx2 - X1 - 1, by2 - Y1 - 1), (220, 0, 0), 1)
    return img


def _png(arr) -> bytes:
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def summary_markdown(diag: dict) -> str:
    c, w = diag["counts"], diag["warning_counts"]
    lines = [f"# Diagnostic run {diag['created']}", "",
             f"- Input: `{diag['input']['name']}`, {diag['input']['width']} x {diag['input']['height']} px, SHA-256 `{diag['input']['sha256']}`",
             f"- Model: {diag['model'].get('checkpoint_label')} `{diag['model'].get('checkpoint_path')}`, SHA-256 `{diag['model'].get('checkpoint_sha256')}`, device {diag['model'].get('device')}",
             f"- Code fingerprint: `{diag['code']['sha256']}` ({diag['code']['files']} files; {diag['code']['wordocr_version']}, {diag['code']['pipeline_version']}, {diag['code']['diagnostics_version']})",
             f"- Settings: {diag['settings']}; crop margin {diag['margin_px']} px",
             f"- Layout: {diag['segmentation']['blocks']} block(s), {diag['segmentation']['lines']} line(s), {c['regions']} region(s); text height {diag['segmentation']['text_height_px']} px; skew {diag['segmentation']['skew_deg']} deg",
             f"- Read: {c['ok']} ok, {c['empty']} empty, {c['failed']} failed; {diag['regions_with_any_warning']} region(s) with at least one warning",
             "", "## Warnings (heuristics, not verdicts)", ""]
    lines += [f"- `{k}`: {n} - {diag['warning_rules'][k]}" for k, n in w.items() if n]
    lines += ["", "## Segmentation notes", ""] + [f"- {f}" for f in diag["segmentation"].get("flags", [])] or ["- none"]
    lines += ["", "No reference text is known for this page: nothing here is an accuracy figure.", ""]
    return "\n".join(lines)


def bundle_files(decoded, seg, result, settings, diag: dict, include_input: bool = True, input_bytes: bytes | None = None) -> dict:
    """{path in the bundle: bytes}. Crops are re-cut with the recognizer's own function and checked against the hashes in `diag`."""
    files = {}
    crops = pipeline.ocr_crops(seg.ocr, seg.words, int(diag["margin_px"]))
    for r, crop in zip(diag["regions"], crops):
        if hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest() != r["crop_sha256"]:
            raise ValueError(f"crop of region {r['id']} differs from the analysed one")
        r = dict(r)
        files[f"crops/{r['number']:04d}_{r['id']}.png"] = _png(crop)
    meta = dict(diag)
    meta["regions"] = [dict(r, crop_file=f"crops/{r['number']:04d}_{r['id']}.png") for r in diag["regions"]]
    meta["input"] = dict(diag["input"], included_in_bundle=bool(include_input and input_bytes is not None))
    files["metadata.json"] = json.dumps(meta, indent=1, ensure_ascii=False).encode("utf-8")
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=REGION_COLUMNS + ["crop_file"])
    wr.writeheader()
    for row, r in zip(region_rows(diag), meta["regions"]):
        wr.writerow(dict(row, crop_file=r["crop_file"]))
    files["regions.csv"] = ("﻿" + buf.getvalue()).encode("utf-8")
    files["lines.json"] = json.dumps(diag["lines"], indent=1).encode("utf-8")
    files["page_text.txt"] = diag["page_text"].encode("utf-8")
    files["overlay_ocr_frame.png"] = _png(overlay_ocr(seg, diag))
    files["overlay_original_frame.png"] = _png(overlay_original(decoded, diag, seg.h))
    geo = diag.get("line_geometry")
    if geo is not None:
        files["line_geometry.json"] = json.dumps(geo, indent=1).encode("utf-8")
        if "error" not in geo:
            files["overlay_original_frame_angle_lines.png"] = _png(overlay_original(decoded, diag, seg.h, "angle", seg))
    if include_input and input_bytes is not None:
        if hashlib.sha256(input_bytes).hexdigest() != decoded.sha256:
            raise ValueError("the input bytes do not belong to this run")
        files["input/" + Path(decoded.name).name.replace("..", "_")] = input_bytes
    files["SUMMARY.md"] = summary_markdown(diag).encode("utf-8")
    files["SHA256SUMS.txt"] = "".join(f"{hashlib.sha256(b).hexdigest()} *{p}\n" for p, b in sorted(files.items())).encode("ascii")
    return files


def bundle_zip(decoded, seg, result, settings, diag: dict, include_input: bool = True, input_bytes: bytes | None = None) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p, b in bundle_files(decoded, seg, result, settings, diag, include_input, input_bytes).items():
            z.writestr(p, b)
    return out.getvalue()


def save_bundle(folder, decoded, seg, result, settings, diag: dict, include_input: bool = True, input_bytes: bytes | None = None) -> Path:
    """Write the bundle into a NEW folder (never overwrites). Refuses protected and read-only project folders."""
    from pageocr import assert_path_allowed
    folder = assert_path_allowed(Path(folder), write=True)
    if folder.exists():
        raise FileExistsError(f"{folder.name} exists; a diagnostic bundle is never overwritten")
    for p, b in bundle_files(decoded, seg, result, settings, diag, include_input, input_bytes).items():
        q = folder / p
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(b)
    return folder
