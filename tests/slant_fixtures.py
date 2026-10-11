"""
Synthetic Tamil pages whose text lines have KNOWN geometry: each line its own angle, optional uneven spacing, notebook
ruling under or through the letters, a curved baseline, or a perspective warp of the whole page.

    page = make_slanted_page(angles=[0, 1.5, -1.5, 3], seed=1)
    page.gray          2-D uint8 page
    page.lines         [TruthLine(index, angle_deg, centre=(x, y), p0, p1, curved, words)] top to bottom
    page.words         [(text, (x1, y1, x2, y2))] axis-aligned boxes of the transformed word ink, in reading order
    page.text          words joined by spaces, lines by newlines

Angles are visual degrees: positive = the line rises to the right (counter-clockwise), the convention of
pageocr.line_geometry. Each line is rendered as a horizontal strip with the project's renderer (tests/fullpage_fixtures)
and rotated about its centre into place; truth boxes and line end points are the transformed coordinates. Generated in
memory: no file, dataset or protected folder is read. Labels come from the generator, never from a model.
"""
from dataclasses import dataclass

import cv2
import numpy as np

import fullpage_fixtures as fx


@dataclass(frozen=True)
class TruthLine:
    index: int
    angle_deg: float | None     # None for a curved line (no single angle)
    centre: tuple
    p0: tuple                   # left and right end of the line's centre line, in page pixels
    p1: tuple
    curved: bool
    words: tuple                # indices into page.words


@dataclass(frozen=True)
class SlantedPage:
    gray: np.ndarray
    lines: tuple
    words: tuple
    text: str


def _strip(rng, width, size, font, rule, curve_amp):
    """One horizontal line of words: coverage strip, word boxes (strip coords), centre-line y, words."""
    sh = int(size * 2.4)
    cover = np.zeros((sh, width), np.float32)
    base = int(size * 1.55)                       # common baseline
    x, words = int(size * 0.6), []
    while True:
        t = fx.WORDS[int(rng.integers(len(fx.WORDS)))]
        m = fx.render_word(t, font, size).astype(np.float32) / 255.0
        mh, mw = m.shape
        if x + mw > width - int(size * 0.6):
            break
        dy = int(round(curve_amp * np.sin(np.pi * (x + mw / 2) / width))) if curve_amp else 0
        top = base - int(mh * 0.62) - size // 2 + dy
        top = max(0, min(sh - mh, top))
        region = cover[top:top + mh, x:x + mw]
        np.maximum(region, m, out=region)
        ys, xs = np.nonzero(m > 0.25)
        words.append((t, (x + int(xs.min()), top + int(ys.min()), x + int(xs.max()) + 1, top + int(ys.max()) + 1)))
        x += mw + int(size * 0.45 * float(rng.uniform(0.85, 1.25)))
    if not words:
        raise ValueError("strip too narrow")
    ys = [b[1] for _, b in words] + [b[3] for _, b in words]
    lo, hi = min(ys), max(ys)
    if rule == "below":
        cv2.line(cover, (0, hi - 1), (width - 1, hi - 1), 0.55, 2)
    elif rule == "through":
        cv2.line(cover, (0, int(lo + 0.72 * (hi - lo))), (width - 1, int(lo + 0.72 * (hi - lo))), 0.55, 2)
    return cover, words, (lo + hi) / 2.0, words[-1][1][2]


def make_slanted_page(angles, seed: int = 0, size: int = 32, width: int = 1500, margin: int = 80, pitch=None,
                      font: str = "NotoSansTamil-Regular", rule: str | None = None, curve_amp: float = 0.0,
                      perspective: float = 0.0, noise: float = 3.0, ink: int = 30, paper: int = 238) -> SlantedPage:
    """angles: one angle per line (degrees). pitch: one centre-to-centre distance per gap (pixels), default 2.3 x size.
    curve_amp: vertical bow of each line in pixels (0 = straight). perspective: vertical shrink of the page's right edge
    at top and bottom, as a fraction of the height (0 = none), so the lines converge to the right with different angles;
    applied to the finished page, truth transformed with it."""
    rng = np.random.default_rng(seed)
    n = len(angles)
    pitch = list(pitch) if pitch is not None else [int(size * 2.3)] * max(0, n - 1)
    strip_w = width - 2 * margin
    height = int(margin * 2 + sum(pitch) + size * 3 + abs(np.tan(np.radians(max(map(abs, angles), default=0)))) * strip_w)
    cover = np.zeros((height, width), np.float32)
    y = margin + size * 1.2 + abs(np.tan(np.radians(max(map(abs, angles), default=0)))) * strip_w / 2
    words, lines = [], []
    for li, ang in enumerate(angles):
        s, ws, cy, xend = _strip(rng, strip_w, size, font, rule, curve_amp)
        cxs = strip_w / 2.0
        m = cv2.getRotationMatrix2D((cxs, cy), float(ang), 1.0)       # counter-clockwise about the strip centre
        m[0, 2] += margin
        m[1, 2] += y - cy
        cover = np.maximum(cover, cv2.warpAffine(s, m, (width, height), flags=cv2.INTER_LINEAR, borderValue=0))
        idx = []
        for t, (x1, y1, x2, y2) in ws:
            pts = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], float) @ m.T
            words.append((t, (int(np.floor(pts[:, 0].min())), int(np.floor(pts[:, 1].min())), int(np.ceil(pts[:, 0].max())), int(np.ceil(pts[:, 1].max())))))
            idx.append(len(words) - 1)
        x0 = ws[0][1][0]
        e = np.array([[x0, cy, 1], [xend, cy, 1], [(x0 + xend) / 2, cy, 1]], float) @ m.T
        lines.append(TruthLine(li, None if curve_amp else float(ang), tuple(e[2]), tuple(e[0]), tuple(e[1]), bool(curve_amp), tuple(idx)))
        if li < len(pitch):
            y += pitch[li]
    img = paper - cover * (paper - ink)
    img = cv2.GaussianBlur(img, (3, 3), 0.6) + rng.normal(0, noise, img.shape)
    gray = np.clip(img, 0, 255).astype(np.uint8)
    if perspective:
        src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        d = perspective * height                   # the right edge shrinks vertically: the lines fan out, each its own angle
        dst = np.float32([[0, 0], [width, d], [width, height - d], [0, height]])
        P = cv2.getPerspectiveTransform(src, dst)
        gray = cv2.warpPerspective(gray, P, (width, height), flags=cv2.INTER_LINEAR, borderValue=paper)
        tp = lambda pts: cv2.perspectiveTransform(np.array(pts, np.float32).reshape(-1, 1, 2), P).reshape(-1, 2)
        nw = []
        for t, (x1, y1, x2, y2) in words:
            q = tp([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
            nw.append((t, (int(np.floor(q[:, 0].min())), int(np.floor(q[:, 1].min())), int(np.ceil(q[:, 0].max())), int(np.ceil(q[:, 1].max())))))
        words = nw
        nl = []
        for ln in lines:
            q = tp([ln.p0, ln.p1, ln.centre])
            a = float(-np.degrees(np.arctan2(q[1][1] - q[0][1], q[1][0] - q[0][0])))
            nl.append(TruthLine(ln.index, None if ln.curved else a, tuple(q[2]), tuple(q[0]), tuple(q[1]), ln.curved, ln.words))
        lines = nl
    text = "\n".join(" ".join(words[i][0] for i in ln.words) for ln in lines)
    return SlantedPage(gray=gray, lines=tuple(lines), words=tuple(words), text=text)
