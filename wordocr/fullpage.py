"""
Full-page OCR for ordinary, unprotected page images: automatic segmentation into word regions, word-by-word
recognition with the project's word recognizer, and the page text rebuilt from the geometry.

    decoded = inputs.decode_upload(name, data)            # format, size, bit depth, frames, protected-data guard
    seg     = segment_page(decoded, PageSettings())       # blocks, lines, words, reading order (no model)
    result  = recognize_page(model, decoded, seg, PageSettings())   # one PageWord per word region + page text

Everything is the project's existing machinery, unchanged:
    segmentation   pageocr.pipeline.segment (layout image: illumination correction, Sauvola threshold, component
                   cleaning; deskew; columns by gutters; line bands; word groups with marks attached; reading order)
    polarity       a page of light text on a dark background is inverted first (255 - value, wordocr.pagedetect.dark_on_light;
                   exact and recorded), because the segmentation looks for ink darker than the paper
    crops          pageocr.pipeline.ocr_crops: cut from the OCR image (original gray pixels, deskewed if needed),
                   never from the thresholded layout image
    recognition    pageocr.engine.OcrEngine of a registered, hash-verified checkpoint (wordocr.service.load_model);
                   each crop is a batch of one, as in word mode, so a word reads the same alone and inside a page
    guard          wordocr.guard.check_image on the page (file hash AND pixels) right before the first crop is read;
                   pageocr.devguard refuses development pages until their human-verified set is frozen

The text: words of a line joined by one space (a punctuation unit joins the word before it without a space), lines by
a newline, blocks (columns, separate regions) and paragraphs (a clearly larger step between two lines of a block) by a
blank line, in the segmentation's reading order. No dictionary,
language model or correction: every word of the text is the raw prediction of one listed word region.

This is NOT a validated OCR system: the segmentation has been tested on generated pages only, and no accuracy has been
measured on real printed pages.
"""
import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
import torch

from pageocr import devguard, pipeline
from pageocr.engine import InvalidCropError, validate_crop
from pageocr.layout import OcrImage, SegParams
from wordocr import guard, pagedetect
from wordocr.inputs import DecodedImage

DEFAULTS = SegParams()
PARAGRAPH_GAP = 1.6                      # a line step this many times the block's median pitch starts a new paragraph
MIN_TEXT_HEIGHT = 5.0                    # pixels; below this the "text" is noise or far too small for the recognizer
MAX_WORD_REGIONS = 3000                  # a page with more regions is almost certainly not text (noise, a photo)


@dataclass(frozen=True)
class PageSettings:
    """The user-adjustable settings. Each maps onto one pageocr parameter or the crop margin; nothing else changes."""
    margin: str | int = "auto"           # pixels added around every word box when cutting the crop; "auto" = 25 % of the text height (2-16 px)
    sauvola_k: float = DEFAULTS.sauvola_k            # threshold sensitivity: lower finds fainter ink
    gap_floor_h: float = DEFAULTS.gap_floor_h        # smallest word gap, in line heights: higher merges more
    deskew: bool = True                              # measure and correct page skew up to 5 degrees
    # EXPERIMENTAL, opt-in, off by default. With all of them off, everything below behaves exactly as before.
    exp_rule_removal: bool = False                   # segmentation: drop long horizontal rule pixels (pageocr.rules)
    exp_contrast_clahe: bool = False                 # segmentation: local contrast equalisation before thresholding
    exp_crop_rule_fill: bool = False                 # recognition: crops get rule pixels replaced by paper (changes model input)
    exp_angle_lines: bool = False                    # segmentation: lines followed along their own direction (pageocr.slant_lines)

    def seg_params(self) -> SegParams:
        return replace(DEFAULTS, sauvola_k=float(self.sauvola_k), gap_floor_h=float(self.gap_floor_h),
                       deskew_max_deg=DEFAULTS.deskew_max_deg if self.deskew else 0.0,
                       exp_rule_removal=bool(self.exp_rule_removal), exp_contrast_clahe=bool(self.exp_contrast_clahe),
                       exp_angle_lines=bool(self.exp_angle_lines))

    def experimental(self) -> dict:
        return {k: v for k, v in (("exp_rule_removal", self.exp_rule_removal), ("exp_contrast_clahe", self.exp_contrast_clahe),
                                  ("exp_crop_rule_fill", self.exp_crop_rule_fill), ("exp_angle_lines", self.exp_angle_lines)) if v}

    def margin_px(self, text_height: float) -> int:
        if self.margin == "auto":
            return int(np.clip(round(0.25 * float(text_height or 0)), 2, 16))
        return int(np.clip(int(self.margin), 0, 64))

    def as_dict(self) -> dict:
        d = {"margin": self.margin, "sauvola_k": self.sauvola_k, "gap_floor_h": self.gap_floor_h, "deskew": self.deskew}
        d.update(self.experimental())                # experimental keys appear only when switched on
        return d


class PageError(ValueError):
    """The page cannot be processed. str(error) is a complete, user-facing message."""


@dataclass(frozen=True)
class PageWord:
    index: int                # 0-based position in reading order
    id: str                   # "b00_l003_w02": block, line in block, word in line
    block: int
    line: int
    word: int
    kind: str                 # "word" or "punctuation"
    box: tuple                # (x1, y1, x2, y2) exclusive, ORIGINAL page pixels (as uploaded)
    box_ocr: tuple            # the same region in the deskewed image the crop is cut from
    ok: bool
    text: str = ""            # raw prediction; "" if not ok or if the model emitted nothing
    error: str = ""
    uncalibrated_score: float | None = None
    crop_width: int = 0
    crop_height: int = 0
    seconds: float | None = None

    @property
    def status(self) -> str:
        return "failed" if not self.ok else ("empty" if self.text == "" else "ok")


@dataclass
class PageResult:
    name: str
    sha256: str
    width: int
    height: int
    words: list
    text: str
    settings: dict
    margin_px: int
    segmentation: dict                     # skew, text height, flags, counts, steps, removed components
    model: dict
    seconds: dict = field(default_factory=dict)

    @property
    def counts(self) -> dict:
        return {"regions": len(self.words), "ok": sum(w.status == "ok" for w in self.words),
                "empty": sum(w.status == "empty" for w in self.words), "failed": sum(w.status == "failed" for w in self.words)}


def segment_page(decoded: DecodedImage, settings: PageSettings = PageSettings()):
    """The segmentation of an uploaded page (pageocr.pipeline.Segmentation). No recognizer is involved."""
    gray, inverted = pagedetect.dark_on_light(decoded.gray)     # light text on a dark page is inverted (exact, reversible)
    try:
        seg = pipeline.segment(OcrImage(gray=gray, page_sha256=decoded.sha256), settings.seg_params())
    except Exception as e:
        raise PageError(f"The page could not be segmented ({type(e).__name__}).") from None
    if seg.words and seg.h < MIN_TEXT_HEIGHT:
        raise PageError(f"The text found on this image is only about {seg.h:.0f} px tall, too small to read (the model needs "
                        "roughly 15 px or more), or the image holds no printed text. Scan or photograph the page at a higher "
                        "resolution.")
    if len(seg.words) > MAX_WORD_REGIONS:
        raise PageError(f"The segmentation found {len(seg.words)} regions, more than the limit of {MAX_WORD_REGIONS}. "
                        "The image is probably not a page of printed text (or is very noisy).")
    object.__setattr__(seg, "polarity_inverted", inverted)    # recorded with the result (Segmentation is a frozen dataclass)
    return seg


def segmentation_summary(seg) -> dict:
    return {"pipeline_version": pipeline.PIPELINE_VERSION, "polarity_inverted": bool(getattr(seg, "polarity_inverted", False)), "skew_deg": seg.skew["angle_deg"], "deskew_applied": seg.skew["applied"],
            "residual_skew_deg": seg.skew["residual_deg"], "text_height_px": round(float(seg.h), 2), "flags": list(seg.flags),
            "blocks": len(seg.blocks), "lines": sum(len(ls) for ls in seg.lines), "word_regions": len(seg.words),
            "steps": list(seg.steps), "removed_components": dict(seg.removed),
            "parameters": {k: getattr(seg.params, k) for k in seg.params.__dataclass_fields__
                           if not k.startswith("exp_") or getattr(seg.params, k)}}       # experimental options listed only when on


def page_text(words) -> str:
    """Words joined by spaces within a line, punctuation units attached, lines by newlines, paragraphs and blocks by a
    blank line. A paragraph break is geometric: inside a block, a step from one line to the next that is at least
    PARAGRAPH_GAP times the block's median line pitch (blocks of fewer than 3 lines have no pitch to compare with)."""
    blocks: dict = {}
    for w in words:
        line = blocks.setdefault(w.block, {}).setdefault(w.line, {"y": [], "text": []})
        line["y"].append((w.box_ocr[1] + w.box_ocr[3]) / 2.0)               # every region places its line, read or not
        if not w.ok or not w.text:
            continue
        if w.kind == "punctuation" and line["text"]:
            line["text"][-1] += w.text
        else:
            line["text"].append(w.text)
    out = []
    for _, lines in sorted(blocks.items()):
        rows = [(float(np.median(v["y"])), " ".join(v["text"])) for _, v in sorted(lines.items())]
        steps = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
        pitch = float(np.median(steps)) if len(steps) >= 2 else 0.0
        text = ""
        for i, (_, t) in enumerate(rows):
            if not t:
                continue
            if text:
                text += "\n\n" if pitch > 0 and steps[i - 1] >= PARAGRAPH_GAP * pitch else "\n"
            text += t
        if text:
            out.append(text)
    return "\n\n".join(out)


def recognize_page(model, decoded: DecodedImage, seg, settings: PageSettings = PageSettings(), progress=None) -> PageResult:
    """Recognize every word region of `seg` with `model` (wordocr.service.LoadedModel). One failed region never stops
    the others. `progress(done, total)` is called after each region if given."""
    # the guard, on the page as uploaded (file hash) and on its pixels, before any crop reaches the model
    guard.check_image(decoded.sha256, decoded.gray)
    devguard.assert_ocr_allowed(seg.ocr.page_sha256)
    t0 = time.perf_counter()
    margin = settings.margin_px(seg.h)
    crops = crop_images(seg, settings)
    out = []
    for i, (pw, crop) in enumerate(zip(seg.words, crops)):
        base = dict(index=i, id=pw.id, block=pw.block_index, line=pw.line_index, word=pw.word_index, kind=pw.kind,
                    box=tuple(int(v) for v in pw.box_original), box_ocr=tuple(int(v) for v in pw.box),
                    crop_width=int(crop.shape[1]), crop_height=int(crop.shape[0]))
        try:
            validate_crop(crop)
        except InvalidCropError as e:
            out.append(PageWord(ok=False, error=f"crop not readable: {e}", **base))
        else:
            t = time.perf_counter()
            try:
                with model.lock, torch.inference_mode():
                    rec = model.engine.recognize_batch([crop], batch_size=1)[0]
                out.append(PageWord(ok=True, text=rec.text, uncalibrated_score=round(rec.confidence, 4),
                                    seconds=round(time.perf_counter() - t, 4), **base))
            except Exception as e:
                out.append(PageWord(ok=False, error=f"recognition failed ({type(e).__name__})", **base))
        if progress is not None:
            progress(i + 1, len(crops))
    return PageResult(name=decoded.name, sha256=decoded.sha256, width=decoded.width, height=decoded.height, words=out,
                      text=page_text(out), settings=settings.as_dict(), margin_px=margin, segmentation=segmentation_summary(seg),
                      model=model.info(), seconds={"recognition": round(time.perf_counter() - t0, 3)})


def overlay(seg, words=None, show_lines: bool = True) -> np.ndarray:
    """RGB picture of the image the crops are cut from (deskewed if it was) with blocks (blue), lines (green) and the
    word regions (red, numbered in reading order from 1). For viewing only."""
    img = cv2.cvtColor(seg.ocr.gray, cv2.COLOR_GRAY2RGB)
    scale = max(0.35, min(1.2, seg.h / 30.0)) if seg.h else 0.5
    thick = max(1, int(round(seg.h / 25.0))) if seg.h else 1
    for b in seg.blocks:
        cv2.rectangle(img, (b.x1 - 3, b.y1 - 3), (b.x2 + 2, b.y2 + 2), (40, 90, 220), thick + 1)
    if show_lines:
        for b, ls in zip(seg.blocks, seg.lines):
            for ln in ls:
                cv2.rectangle(img, (b.x1, ln.y1), (b.x2 - 1, ln.y2 - 1), (0, 150, 0), 1)
    status = {w.index: w.status for w in words} if words else {}
    for i, w in enumerate(seg.words):
        x1, y1, x2, y2 = w.box
        colour = (200, 0, 0) if status.get(i, "ok") == "ok" else (230, 120, 0)
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), colour, thick)
        cv2.putText(img, str(i + 1), (x1, max(int(10 * scale) + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale,
                    (200, 0, 0), max(1, thick - 1), cv2.LINE_AA)
    return img


def crop_images(seg, settings: PageSettings = PageSettings()) -> list:
    """The exact crops the recognizer receives (for inspection and for recognize_page)."""
    crops = pipeline.ocr_crops(seg.ocr, seg.words, settings.margin_px(seg.h))
    if not settings.exp_crop_rule_fill:
        return crops
    # EXPERIMENTAL: fill rule pixels with paper
    from pageocr.rules import fill_rules, rule_pixels
    from pageocr.layout import sauvola, correct_illumination
    norm = correct_illumination(seg.ocr.gray, seg.params.illumination_kernel_frac)
    window = max(25, int(round(seg.params.sauvola_window_h * seg.h)) | 1)
    ink = sauvola(norm, window, seg.params.sauvola_k, seg.params.sauvola_r, seg.params.min_contrast)
    filled = fill_rules(seg.ocr.gray, rule_pixels(ink, float(seg.h)))
    from pageocr.layout import OcrImage
    return pipeline.ocr_crops(OcrImage(gray=filled, page_sha256=seg.ocr.page_sha256, deskew_deg=seg.ocr.deskew_deg,
                                       to_original=seg.ocr.to_original, original_shape=seg.ocr.original_shape),
                              seg.words, settings.margin_px(seg.h))
