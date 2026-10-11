"""
EXPERIMENTAL (opt-in, off by default): horizontal rule lines (notebook ruling, underlines) in an ink mask.

The default segmentation already drops a rule line that stands free as its own component (pageocr.layout, "rule
line"). It cannot drop a rule that TOUCHES text: rule and letters then form one component, which is either kept as a
giant "word" or dropped as a "figure" together with the text it touches. This module finds the rule pixels themselves:

    rules = rule_pixels(ink, h)        # bool mask: long horizontal runs of ink, at least min_len_h x h long
    clean = remove_rules(ink, h)       # ink without the rule pixels, with crossing strokes kept (see below)

A Tamil letter can contain a horizontal stroke, so only runs much longer than a letter are treated as rules
(default 4 x text height, a full word or more of unbroken horizontal ink, which print does not have). Where a rule
crosses a letter stroke, the pixels of the rule that have ink directly above AND below are kept, so a stroke is not cut
in two. Remaining risk, measured rather than assumed: a rule that runs exactly along the bottom of letters (a baseline
rule) can take off the bottom pixel rows of those letters.
"""
import cv2
import numpy as np


def rule_pixels(ink: np.ndarray, h: float, min_len_h: float = 4.0, max_thick_h: float = 0.25,
                min_len_frac: float = 1 / 12, min_thick_px: int = 4) -> np.ndarray:
    """Pixels of horizontal runs at least max(min_len_h x h, min_len_frac x page width) long and at most
    max(max_thick_h x h, min_thick_px) thick.

    rules-1.1 (2026-10-10): the page-width and pixel floors were added after a development page on which ruling and a
    margin line joined all letters into one component; the text height estimated from the remaining fragments was 5 px
    instead of about 34, so 2-3 px rules counted as "thick" and were kept. With the floors the test no longer depends
    on that estimate being right."""
    if not ink.any() or h <= 0:
        return np.zeros(ink.shape, bool)
    L = max(31, int(round(min_len_h * h)), int(ink.shape[1] * min_len_frac))
    m = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))).astype(bool)
    if not m.any():
        return m
    # keep only thin structures: a filled block of text rows opened horizontally is not a rule
    t = max(min_thick_px, int(round(max_thick_h * h)))
    thick = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, t + 1))).astype(bool)
    return m & ~thick


def remove_rules(ink: np.ndarray, h: float, **kw) -> tuple:
    """(ink without rule pixels, the rule mask). Rule pixels with ink directly above and below (a crossing stroke) stay."""
    rules = rule_pixels(ink, h, **kw)
    if not rules.any():
        return ink, rules
    t = max(3, int(round(0.3 * h)))
    above = np.zeros_like(ink)
    below = np.zeros_like(ink)
    rest = ink & ~rules
    # nearest non-rule ink within t rows above / below each pixel
    for d in range(1, t + 1):
        above[d:, :] |= rest[:-d, :]
        below[:-d, :] |= rest[d:, :]
    crossing = rules & above & below
    return rest | crossing, rules & ~crossing


def fill_rules(gray: np.ndarray, rules: np.ndarray) -> np.ndarray:
    """A copy of gray with the rule pixels replaced by the paper level around them (for the opt-in crop option)."""
    if not rules.any():
        return gray
    out = gray.copy()
    grown = cv2.dilate(rules.astype(np.uint8), np.ones((3, 1), np.uint8)).astype(bool)
    paper = cv2.medianBlur(gray, 9)
    bright = cv2.dilate(paper, np.ones((7, 1), np.uint8))          # the lighter of the rows around: paper, not the rule
    out[grown] = bright[grown]
    return out
