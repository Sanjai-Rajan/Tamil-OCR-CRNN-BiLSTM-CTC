"""
Synthetic test pages under controlled image conditions, with trustworthy labels: the generator knows every word it
typeset and the box it put it in (tests/fullpage_fixtures.make_tamil_page). Geometric conditions (skew, perspective)
transform the label boxes with the same matrix as the image. Photometric conditions (ruling, shadow, low contrast,
blur) leave the boxes unchanged.

These pages are rendered with the fonts and the generator the model was trained with. They can show whether the
SEGMENTATION keeps words intact under each condition, and whether a change helps or hurts. They are not evidence of
accuracy on real documents, and in particular not on real handwriting: "handwriting_style_font" is a printed font
that imitates handwriting.

    page = make(condition, seed)   ->  SyntheticPage(gray, words=[(text, (x1, y1, x2, y2)), ...], text, condition, seed)
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import fullpage_fixtures as fx  # noqa: E402

CONDITIONS = {
    "clean": "one column, default print",
    "two_columns": "two columns with a heading",
    "ruled_below": "notebook ruling: a gray rule under every line, touching the lowest strokes",
    "ruled_through": "ruling through the lower part of the letters (rules misaligned with the writing)",
    "low_contrast": "ink 150 on paper 205",
    "shadow": "illumination falling from 100 % to 45 % across the page",
    "blur": "Gaussian blur, sigma 1.6",
    "tight_spacing": "word spaces about 40 % of normal",
    "small_print": "text height about 20 px",
    "skew_3deg": "page rotated by 3.5 degrees",
    "perspective": "trapezoid: top edge 8 % narrower than the bottom",
    "handwriting_style_font": "printed font Kavivanar (imitates handwriting; NOT handwriting)",
}


@dataclass
class SyntheticPage:
    gray: np.ndarray
    words: list
    text: str
    condition: str
    seed: int


def _transform_boxes(words, M, persp=False):
    out = []
    for t, (x1, y1, x2, y2) in words:
        pts = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], np.float32).reshape(-1, 1, 2)
        q = cv2.perspectiveTransform(pts, M) if persp else cv2.transform(pts, M)
        q = q.reshape(-1, 2)
        out.append((t, (int(np.floor(q[:, 0].min())), int(np.floor(q[:, 1].min())), int(np.ceil(q[:, 0].max())), int(np.ceil(q[:, 1].max())))))
    return out


def make(condition: str, seed: int) -> SyntheticPage:
    if condition not in CONDITIONS:
        raise KeyError(condition)
    kw = dict(seed=seed, columns=1, lines=9)
    if condition == "two_columns":
        kw.update(columns=2, heading=True, lines=8)
    elif condition == "tight_spacing":
        kw.update(word_space=0.17)
    elif condition == "small_print":
        kw.update(size=20)
    elif condition == "low_contrast":
        kw.update(ink=150, paper=205)
    elif condition == "handwriting_style_font":
        kw.update(font="Kavivanar-Regular")
    p = fx.make_tamil_page(**kw)
    gray = p.gray.astype(np.float32)
    words = [(w.text, tuple(w.box)) for w in p.words]
    H, W = gray.shape
    rng = np.random.default_rng(seed + 7)
    if condition in ("ruled_below", "ruled_through"):
        # one rule per text line; its position relative to the ink of the line
        by_line = {}
        for w in p.words:
            by_line.setdefault((w.block, w.line), []).append(w.box)
        for boxes in by_line.values():
            top = min(b[1] for b in boxes)
            bot = max(b[3] for b in boxes)
            y = int(bot - 1) if condition == "ruled_below" else int(top + 0.72 * (bot - top))
            cv2.line(gray, (0, y), (W - 1, y), float(rng.uniform(120, 150)), 2)
        x_margin = int(0.04 * W)
        cv2.line(gray, (x_margin, 0), (x_margin, H - 1), 140.0, 2)       # the notebook's margin line
    elif condition == "shadow":
        ramp = np.linspace(1.0, 0.45, W, dtype=np.float32)[None, :]
        gray = gray * ramp
    elif condition == "blur":
        gray = cv2.GaussianBlur(gray, (0, 0), 1.6)
    elif condition == "skew_3deg":
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 3.5, 1.0)
        gray = cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=238)
        words = _transform_boxes(words, M)
    elif condition == "perspective":
        d = 0.04 * W
        src = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
        dst = np.float32([[d, 0], [W - d, 0], [0, H], [W, H]])
        M = cv2.getPerspectiveTransform(src, dst)
        gray = cv2.warpPerspective(gray, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=238)
        words = _transform_boxes(words, M, persp=True)
    g = np.clip(gray, 0, 255).astype(np.uint8)
    return SyntheticPage(gray=g, words=words, text=p.text, condition=condition, seed=seed)
