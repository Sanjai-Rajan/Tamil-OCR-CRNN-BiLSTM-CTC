"""
Generated pages with known word boxes, for testing the V3 segmentation. Not a test file itself.

The "print" is made of invented glyphs (strokes, loops, dots), not of any script: these pages test geometry - skew,
columns, lines, word gaps, marks, reading order - and say nothing about recognition. They are engineering fixtures,
never evidence of OCR accuracy, and they are built in memory; nothing is read from the project's data.

    page = make_page(layout="two_columns", seed=3, skew_deg=2.0, uneven_light=True)
    page.gray     2-D uint8, not binarized (noise and blur give it many gray levels)
    page.words    [{"box": (x1, y1, x2, y2), "block": b, "line": l, "word": w}] in reading order, ORIGINAL page pixels
    page.blocks   number of blocks in reading order;  page.lines  number of lines
"""
from dataclasses import dataclass

import cv2
import numpy as np

PAPER, INK = 232, 38


@dataclass
class SynthPage:
    gray: np.ndarray
    words: list
    blocks: int
    lines: int
    spec: dict


def _glyph(rng, h: int) -> np.ndarray:
    """One connected glyph, a boolean array about h high and 0.5-0.8 h wide."""
    w = int(h * rng.uniform(0.5, 0.8))
    g = np.zeros((h, w), np.uint8)
    t = max(2, h // 9)
    kind = rng.integers(0, 4)
    if kind == 0:                                              # loop with a stem
        cv2.ellipse(g, (w // 2, h // 2), (w // 2 - t, h // 2 - t), 0, 0, 360, 1, t)
        cv2.line(g, (w - t, t), (w - t, h - 1), 1, t)
    elif kind == 1:                                            # two stems and a bar
        cv2.line(g, (t, t), (t, h - 1), 1, t); cv2.line(g, (w - t, t), (w - t, h - 1), 1, t); cv2.line(g, (t, h // 2), (w - t, h // 2), 1, t)
    elif kind == 2:                                            # hook
        cv2.ellipse(g, (w // 2, h // 3), (w // 2 - t, h // 3 - t), 0, 180, 360, 1, t)
        cv2.line(g, (w - t, h // 3), (w - t, h - 1), 1, t); cv2.line(g, (t, h - t), (w - t, h - t), 1, t)
    else:                                                      # zigzag
        cv2.polylines(g, [np.array([[t, t], [w - t, t], [t, h - t], [w - t, h - t]], np.int32)], False, 1, t)
    return g.astype(bool)


def _put(mask: np.ndarray, glyph: np.ndarray, x: int, y: int) -> tuple:
    h, w = glyph.shape
    H, W = mask.shape
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return None
    mask[y:y + h, x:x + w] |= glyph
    ys, xs = np.nonzero(glyph)
    return (x + int(xs.min()), y + int(ys.min()), x + int(xs.max()) + 1, y + int(ys.max()) + 1)


def _union(a, b):
    return b if a is None else (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _set_line(rng, mask, x0, x1, y, h, word_gap, letter_gap, marks, punctuation, justify, solid=False):
    """Words from x0 to x1 on a line whose letter tops are at y. Returns their tight boxes."""
    plans, x = [], x0
    while True:                                                # plan the words first, so the line can be justified
        n = int(rng.integers(2, 7))
        glyphs = [_glyph(rng, h) for _ in range(n)]
        width = sum(g.shape[1] for g in glyphs) + letter_gap * (n - 1)
        if x + width > x1:
            break
        plans.append(glyphs)
        x += width + word_gap
    if not plans:
        return []
    widths = [sum(g.shape[1] for g in gl) + letter_gap * (len(gl) - 1) for gl in plans]
    gap = word_gap
    if solid:
        gap = letter_gap
    elif justify and len(plans) > 1:
        gap = min(int(1.6 * word_gap), max(word_gap, (x1 - x0 - sum(widths)) // (len(plans) - 1)))     # stretched, within what a compositor would allow
    boxes, x = [], x0
    for gl in plans:
        box = None
        for g in gl:
            b = _put(mask, g, x, y)
            if b is None:
                break
            box = _union(box, b)
            if marks and rng.random() < 0.3:                   # a pulli: a dot above the letter, clear of it
                r = max(2, h // 9)
                dot = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
                cv2.circle(dot, (r, r), r, 1, -1)
                d = _put(mask, dot.astype(bool), x + g.shape[1] // 2 - r, y - 3 * r - 3)
                box = _union(box, d) if d else box
            elif marks and rng.random() < 0.12:                # a superscript mark beside the letter's top
                r = max(2, h // 8)
                m = np.zeros((2 * r, r + 1), np.uint8)
                cv2.line(m, (r // 2, 0), (r // 2, 2 * r - 1), 1, 2)
                d = _put(mask, m.astype(bool), x + g.shape[1] + 1, y - r)
                box = _union(box, d) if d else box
            x += g.shape[1] + letter_gap
        x -= letter_gap
        if punctuation and rng.random() < 0.3:                 # a full stop or comma close after the word, low on the line
            r = max(2, h // 10)
            dot = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
            cv2.circle(dot, (r, r), r, 1, -1)
            d = _put(mask, dot.astype(bool), x + 2, y + h - 2 * r - 1)
            if d:
                box, x = _union(box, d), d[2]
        if box:
            boxes.append(box)
        x += gap
    if solid and boxes:                                        # a line set solid is one run
        b = boxes[0]
        for o in boxes[1:]:
            b = _union(b, o)
        return [b]
    return boxes


def make_page(layout: str = "one_column", seed: int = 0, height: int = 1500, width: int = 1100, h: int = 26, lines: int = 14,
              word_gap: float = 0.75, letter_gap: int = 3, marks: bool = True, punctuation: bool = True, justify: bool = False,
              skew_deg: float = 0.0, uneven_light: bool = False, border: bool = False, bleed_through: bool = False, specks: int = 0,
              touching_lines: bool = False, solid_lines: bool = False) -> SynthPage:
    """layout: "one_column", "two_columns", "heading_two_columns" (a full-width heading above two columns)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((height, width), bool)
    margin, pitch, gap = 110, int(h * 2.1), max(letter_gap + 4, int(word_gap * h))
    words, y, block, n_lines = [], margin, 0, 0

    def column(x0, x1, y0, count):
        nonlocal n_lines
        out, yy, li = [], y0, 0
        for _ in range(count):
            if yy + h + 8 > height - margin:
                break
            boxes = _set_line(rng, mask, x0, x1, yy, h, gap, letter_gap, marks, punctuation, justify, solid=solid_lines)
            if boxes:
                out.append((li, boxes)); li += 1; n_lines += 1
            yy += pitch
        return out, yy

    def add(col_lines):
        nonlocal block
        for li, boxes in col_lines:
            for wi, b in enumerate(boxes):
                words.append({"box": b, "block": block, "line": li, "word": wi})
        block += 1

    if layout == "one_column":
        col, _ = column(margin, width - margin, y, lines)
        add(col)
    else:
        if layout == "heading_two_columns":
            head, y = column(margin + 150, width - margin - 150, y, 2)
            add(head)
            y += h
        mid, gutter = width // 2, int(2.6 * h)
        left, _ = column(margin, mid - gutter // 2, y, lines)
        right, _ = column(mid + gutter // 2, width - margin, y, lines - 2)
        add(left); add(right)
    if touching_lines:                                         # one tall stroke that joins the first two lines of the page
        first = [w for w in words if w["block"] == 0 and w["line"] == 0]
        if first:
            b = first[0]["box"]
            mask[b[3] - 2:b[3] - 2 + pitch - h + 6, b[0] + 2:b[0] + 5] = True
    img = np.full((height, width), PAPER, np.float32)
    if bleed_through:                                          # the other side of the leaf: mirrored, faint
        ghost = np.roll(mask[:, ::-1], pitch // 2, axis=0)
        img[ghost] -= 40
    img[mask] = INK
    if border:
        img[:14, :] = 25; img[-14:, :] = 25; img[:, :14] = 25; img[:, -14:] = 25
        cv2.line(img, (60, height - 70), (width - 60, height - 70), 60, 3)       # a printed rule
    for _ in range(specks):
        cv2.circle(img, (int(rng.integers(20, width - 20)), int(rng.integers(20, height - 20))), 1, 70, -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.9)
    if uneven_light:
        gx = np.linspace(1.0, 0.62, width, dtype=np.float32)[None, :]
        gy = np.linspace(1.0, 0.85, height, dtype=np.float32)[:, None]
        img = img * gx * gy
    img = img + rng.normal(0, 3.0, img.shape).astype(np.float32)
    gray = np.clip(img, 0, 255).astype(np.uint8)
    truth = [dict(w) for w in words]
    if skew_deg:
        m = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), skew_deg, 1.0)
        gray = cv2.warpAffine(gray, m, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float(PAPER))
        for w in truth:
            x1, y1, x2, y2 = w["box"]
            pts = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], float) @ m.T
            w["box"] = (int(np.floor(pts[:, 0].min())), int(np.floor(pts[:, 1].min())), int(np.ceil(pts[:, 0].max())), int(np.ceil(pts[:, 1].max())))
    return SynthPage(gray=gray, words=truth, blocks=block, lines=n_lines,
                     spec=dict(layout=layout, seed=seed, h=h, skew_deg=skew_deg, uneven_light=uneven_light, border=border, bleed_through=bleed_through,
                               justify=justify, marks=marks, punctuation=punctuation, word_gap=word_gap, touching_lines=touching_lines, solid_lines=solid_lines))
