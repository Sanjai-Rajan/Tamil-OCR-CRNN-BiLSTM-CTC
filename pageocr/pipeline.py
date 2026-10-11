"""
V3: automatic segmentation of a page into blocks, lines and word boxes, in reading order.

    page -> OCR image (original gray) -> layout image -> skew -> [rotate the OCR image, rebuild the layout image]
         -> blocks (columns) -> lines -> words -> reading order -> word crops cut from the OCR image

    seg = segment(page)                 # no recognizer involved
    crops = ocr_crops(seg.ocr, seg.words, pad_px=0)
    recs = recognize(seg, engine)       # refuses development pages until the human-verified set is frozen

Two image paths (see pageocr.layout): everything that FINDS text works on the layout image; every pixel the
recognizer receives comes from the OCR image. ocr_crops raises TypeError for anything that is not an OcrImage.

STATUS: this is engineering code, tested on generated pages only. It has not been evaluated on any real page.
It is not validated full-page OCR, and no accuracy may be claimed for it.
"""
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np

from pageocr import devguard
from pageocr.columns import find_blocks
from pageocr.deskew import estimate_skew
from pageocr.layout import LAYOUT_VERSION, ImagePathError, LayoutImage, OcrImage, SegParams, build_layout_image
from pageocr.lines import find_lines
from pageocr.reading_order import place, to_boxes
from pageocr.words import find_words, page_threshold

PIPELINE_VERSION = "pageocr-pipeline-3.0.1"     # 3.0.1 (2026-10-10): deskew keeps thin strokes; steps after deskew; arrays identified for the guard
STATEMENT = ("V3 automatic segmentation: classical image processing, tested on generated pages only. "
             "It is not validated full-page OCR, and no accuracy is claimed for it.")


@dataclass(frozen=True)
class Segmentation:
    ocr: OcrImage                 # the image word crops are cut from
    h: float                      # the page's text height in pixels (unit of the parameters)
    skew: dict                    # measured angle, whether it was applied, residual after deskew
    blocks: tuple
    lines: tuple                  # lines[i] = lines of blocks[i]
    words: tuple                  # PlacedWord, in reading order
    word_diagnostics: tuple       # per block, per line: threshold and gaps
    flags: tuple
    removed: dict
    steps: tuple
    params: SegParams
    seconds: float


def _identified(ocr: OcrImage) -> OcrImage:
    """An OCR image without a file hash gets the hash of the development master with exactly its pixels, if there is one,
    so the development guard also sees a development page that arrives as a bare array (pageocr.devguard)."""
    if ocr.page_sha256 or ocr.deskew_deg != 0.0:
        return ocr
    sha = devguard.development_page_for_pixels(ocr.gray)
    return replace(ocr, page_sha256=sha) if sha else ocr


def _as_ocr_image(source) -> OcrImage:
    if isinstance(source, OcrImage):
        return _identified(source)
    if isinstance(source, LayoutImage):
        raise ImagePathError("a layout image cannot be the source of a segmentation's OCR image")
    if hasattr(source, "gray") and hasattr(source, "sha256"):          # pageocr.imageio.Page
        return OcrImage.from_page(source)
    if isinstance(source, np.ndarray):
        return _identified(OcrImage(gray=source))
    raise ImagePathError(f"segment takes a Page, an OcrImage or a 2-D uint8 grayscale array, got {type(source).__name__}")


def segment(source, params: SegParams = SegParams()) -> Segmentation:
    """Blocks, lines and words of a page, in reading order. The recognizer is not involved."""
    t0 = time.perf_counter()
    ocr = _as_ocr_image(source)
    layout = build_layout_image(ocr, params)
    flags, steps = list(layout.flags), list(layout.steps)
    skew = {"angle_deg": 0.0, "applied": False, "residual_deg": 0.0, "sharpness_gain": 1.0}
    if layout.ink.any():
        m = estimate_skew(layout.ink, params.deskew_max_deg)
        skew.update(angle_deg=m["angle_deg"], sharpness_gain=m["sharpness_gain"])
        if abs(m["angle_deg"]) >= params.deskew_min_deg:
            ocr = ocr.rotated(m["angle_deg"])
            layout = build_layout_image(ocr, params)
            flags, steps = list(layout.flags), list(layout.steps)          # describe the layout pass actually used
            skew["applied"] = True
            skew["residual_deg"] = estimate_skew(layout.ink, 2.0)["angle_deg"] if layout.ink.any() else 0.0
            steps.append(f"deskew: OCR image rotated by {m['angle_deg']:+.2f} degrees, layout image rebuilt from it")
            if abs(m["angle_deg"]) >= params.deskew_max_deg - 1e-6:
                flags.append("skew at the limit of the search: the page may be skewed by more than the limit")
            if abs(skew["residual_deg"]) >= 0.8:
                flags.append("lines are not parallel after deskew: curved or non-Manhattan layout suspected")
    blocks, bflags = find_blocks(layout.ink, layout.h, params.gutter_min_h, params.gutter_min_lines) if layout.ink.any() else ([], ["no text found"])
    flags += bflags
    if params.exp_angle_lines:                    # EXPERIMENTAL, off by default: lines followed along their own direction
        lines, words, diags = _angle_lines(layout, blocks, params)
        steps.append("EXPERIMENTAL angle-aware lines: letters followed along each line's own direction; words found on a levelled copy of the box coordinates")
        all_lines = [ln for ls in lines for ln in ls]
    else:
        lines = [find_lines(layout.ink, layout.components, b, params.line_split_factor, params.line_thin_factor) for b in blocks]
        all_lines = [ln for ls in lines for ln in ls]
        pooled = page_threshold(layout.components, all_lines, params) if all_lines else None
        words, diags = [], []
        for ls in lines:
            ws, ds = [], []
            for ln in ls:
                w, d = find_words(layout.components, ln, params, pooled)
                ws.append(w); ds.append(d)
            words.append(ws); diags.append(ds)
    placed = place(blocks, lines, words, ocr)
    n_forced = sum(ln.forced_split for ln in all_lines)
    if n_forced:
        flags.append(f"{n_forced} line(s) were cut out of bands of touching lines; those cuts are guesses")
    if all_lines and sum(1 for ds in diags for d in ds if d["threshold_source"] == "line") < 0.3 * len(all_lines):
        flags.append("few lines had word gaps of two clear sizes: word spacing is uncertain on this page")
    return Segmentation(ocr=ocr, h=layout.h, skew=skew, blocks=tuple(blocks), lines=tuple(tuple(ls) for ls in lines), words=tuple(placed),
                        word_diagnostics=tuple(tuple(ds) for ds in diags), flags=tuple(dict.fromkeys(flags)), removed=dict(layout.removed),
                        steps=tuple(steps), params=params, seconds=time.perf_counter() - t0)


def _angle_lines(layout, blocks, params):
    """EXPERIMENTAL (params.exp_angle_lines): lines, words and word diagnostics from pageocr.slant_lines."""
    from pageocr import slant_lines
    chosen, notes = [], []
    for b in blocks:
        band = find_lines(layout.ink, layout.components, b, params.line_split_factor, params.line_thin_factor)
        tracked = slant_lines.find_lines_tracked(layout.components, b, layout.h)
        use, why = slant_lines.prefer_tracked(layout.components, band, tracked, layout.h)
        chosen.append(tracked if use else [(ln, 0.0, float(b.x1)) for ln in band])
        notes.append(("tracked" if use else "bands", why))
    # blocks that keep their bands go through exactly the default word step (original components, original Line);
    # only tracked blocks use the levelled copy of their own members' boxes
    tracked_only = [t for ts, (kind, _) in zip(chosen, notes) if kind == "tracked" for t in ts]
    lev = slant_lines.levelled_components(layout.components, tracked_only) if tracked_only else layout.components
    for_threshold = [slant_lines.levelled_line(lev, ln) if kind == "tracked" else ln for ts, (kind, _) in zip(chosen, notes) for ln, _, _ in ts]
    pooled = page_threshold(lev, for_threshold, params) if for_threshold else None
    lines, words, diags = [], [], []
    for ts, (kind, why) in zip(chosen, notes):
        ws, ds = [], []
        for ln, slope, _x_ref in ts:
            if kind == "tracked":
                w, d = slant_lines.words_of_tracked(layout.components, lev, ln, params, pooled)
            else:
                w, d = find_words(layout.components, ln, params, pooled)
            ws.append(w); ds.append(dict(d, line_source=kind, line_source_reason=why, line_slope=round(float(slope), 5)))
        lines.append([t[0] for t in ts]); words.append(ws); diags.append(ds)
    return lines, words, diags


def ocr_crops(ocr, words, pad_px: int = 0) -> list:
    """One crop per word, cut from the OCR image: original grayscale, never the layout image.

    pad_px is the margin added on every side at crop time (clipped to the image); the stored boxes stay tight.
    """
    if not isinstance(ocr, OcrImage):
        raise TypeError(f"word crops for the recognizer are cut from an OcrImage only, got {type(ocr).__name__}. "
                        "A layout image, a mask or a plain array is refused here.")
    H, W = ocr.gray.shape
    out = []
    for w in words:
        x1, y1, x2, y2 = w.box
        x1, y1, x2, y2 = max(0, x1 - pad_px), max(0, y1 - pad_px), min(W, x2 + pad_px), min(H, y2 + pad_px)
        out.append(ocr.gray[y1:y2, x1:x2].copy())
    return out


def recognize(seg: Segmentation, engine, pad_px: int = 0, batch_size: int = 32) -> list:
    """One Recognition per usable word of the segmentation, or None where the crop cannot be read (smaller than 2 x 2,
    or two-level). Refuses a development page until the human-verified set is frozen (pageocr.devguard)."""
    if not isinstance(seg, Segmentation):
        raise TypeError("recognize takes a Segmentation")
    devguard.assert_ocr_allowed(seg.ocr.page_sha256)
    if not seg.ocr.page_sha256 and seg.ocr.deskew_deg == 0.0:              # no file hash: identify the pixels themselves
        devguard.assert_ocr_allowed(devguard.development_page_for_pixels(seg.ocr.gray))
    from pageocr.engine import InvalidCropError, validate_crop
    crops = ocr_crops(seg.ocr, seg.words, pad_px)
    good = []
    for i, c in enumerate(crops):
        try:
            validate_crop(c)
            good.append(i)
        except InvalidCropError:
            pass
    recs = engine.recognize_batch([crops[i] for i in good], batch_size=batch_size) if good else []
    out = [None] * len(crops)
    for i, r in zip(good, recs):
        out[i] = r
    return out


def diagnostics(seg: Segmentation) -> dict:
    """Everything about a segmentation except pixels and recognizer output, as plain JSON-able data."""
    words = [{"id": w.id, "block": w.block_index, "line": w.line_index, "word": w.word_index, "kind": w.kind,
              "box": list(w.box), "box_original": list(w.box_original), "gap_before_px": w.gap_before, "line_forced_split": w.line_forced_split}
             for w in seg.words]
    return {
        "statement": STATEMENT, "pipeline_version": PIPELINE_VERSION, "layout_version": LAYOUT_VERSION,
        "page": {"sha256": seg.ocr.page_sha256, "original_height": seg.ocr.original_shape[0], "original_width": seg.ocr.original_shape[1],
                 "ocr_image_height": seg.ocr.shape[0], "ocr_image_width": seg.ocr.shape[1]},
        "image_paths": {"ocr_image": "original grayscale" + (f", rotated by {seg.ocr.deskew_deg:+.2f} degrees" if seg.ocr.deskew_deg else ""),
                        "layout_image": "; ".join(seg.steps)},
        "skew": seg.skew, "text_height_px": seg.h, "flags": list(seg.flags), "removed_components": seg.removed,
        "blocks": [{"index": i, "kind": b.kind, "group": b.group, "column_in_group": b.column_in_group, "box": [b.x1, b.y1, b.x2, b.y2],
                    "lines": [{"index": j, "y1": ln.y1, "y2": ln.y2, "forced_split": ln.forced_split, **seg.word_diagnostics[i][j]} for j, ln in enumerate(seg.lines[i])]}
                   for i, b in enumerate(seg.blocks)],
        "words": words, "counts": {"blocks": len(seg.blocks), "lines": sum(len(l) for l in seg.lines), "words": len(words)},
        "parameters": {k: getattr(seg.params, k) for k in seg.params.__dataclass_fields__}, "seconds": round(seg.seconds, 3),
    }


def draw_overlay(seg: Segmentation, show=("blocks", "lines", "words")) -> np.ndarray:
    """The OCR image in colour with the found blocks (blue), lines (green) and words (red) drawn on it. For viewing only."""
    img = cv2.cvtColor(seg.ocr.gray, cv2.COLOR_GRAY2BGR)
    if "lines" in show:
        for b, ls in zip(seg.blocks, seg.lines):
            for ln in ls:
                cv2.rectangle(img, (b.x1, ln.y1), (b.x2 - 1, ln.y2 - 1), (0, 160, 0), 1)
    if "words" in show:
        for w in seg.words:
            cv2.rectangle(img, (w.box[0], w.box[1]), (w.box[2] - 1, w.box[3] - 1), (0, 0, 220) if w.kind == "word" else (0, 140, 255), 1)
    if "blocks" in show:
        for i, b in enumerate(seg.blocks):
            cv2.rectangle(img, (b.x1 - 2, b.y1 - 2), (b.x2 + 1, b.y2 + 1), (220, 80, 0), 2)
            cv2.putText(img, str(i), (b.x1, max(12, b.y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 80, 0), 2)
    return img


def boxes_for_export(seg: Segmentation, frame: str = "original") -> list:
    return to_boxes(list(seg.words), frame)
