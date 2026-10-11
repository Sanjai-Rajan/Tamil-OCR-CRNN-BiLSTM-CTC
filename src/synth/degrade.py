"""
Image effects that turn a clean coverage mask into a realistic word crop.

modern_print(): clean-to-noisy scans and phone photos of books, newspapers, typed pages.
old_print():    aged books: yellowed/fibrous paper, ink bleed, broken strokes, specks,
                bleed-through from the reverse side, uneven inking, low resolution.

All functions take/return uint8 numpy arrays and a random.Random instance, so results
are reproducible per sample seed.
"""
import random

import cv2
import numpy as np


# ------------------------------------------------------------------ helpers
def _smooth_noise(h, w, rng, scale):
    """Low-frequency noise field in [0,1] (cheap fractal-ish noise via upscaled random grids)."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for octave in range(3):
        gh = max(2, int(h / (scale / (2 ** octave))))
        gw = max(2, int(w / (scale / (2 ** octave))))
        g = np.random.default_rng(rng.getrandbits(32)).random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    out /= total
    return np.clip(out, 0, 1)


def _np_rng(rng):
    return np.random.default_rng(rng.getrandbits(32))


# ------------------------------------------------------------------ geometry
def geometric(mask, rng, strength=1.0):
    h, w = mask.shape
    pad = int(0.25 * h) + int(0.05 * w) + 4
    m = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    H, W = m.shape
    # horizontal stretch (condensed / expanded type)
    sx = rng.uniform(0.85, 1.18)
    m = cv2.resize(m, (max(4, int(W * sx)), H), interpolation=cv2.INTER_LINEAR)
    H, W = m.shape
    pts1 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
    shear = rng.uniform(-0.18, 0.18) * strength if rng.random() < 0.3 else 0.0
    rot = np.deg2rad(rng.uniform(-2.5, 2.5) * strength)
    persp = 0.03 * strength if rng.random() < 0.25 else 0.0
    def jitter(): return rng.uniform(-persp, persp) * H
    cx, cy = W / 2, H / 2
    pts2 = []
    for x, y in pts1:
        x2 = x + shear * (cy - y)
        dx, dy = x2 - cx, y - cy
        xr = cx + dx * np.cos(rot) - dy * np.sin(rot) + jitter()
        yr = cy + dx * np.sin(rot) + dy * np.cos(rot) + jitter()
        pts2.append([xr, yr])
    M = cv2.getPerspectiveTransform(pts1, np.float32(pts2))
    return cv2.warpPerspective(m, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)


def crop_with_margins(mask, rng, tight=False):
    ys, xs = np.where(mask > 20)
    if len(xs) == 0:
        return mask
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h = y1 - y0
    if tight:
        my = [rng.uniform(0.0, 0.08) for _ in range(2)]
        mx = [rng.uniform(0.0, 0.08) for _ in range(2)]
    else:  # mimic segmenter crops: from tight to generous
        my = [rng.uniform(0.0, 0.35) for _ in range(2)]    # never clip ink (a cut vowel sign changes the label)
        mx = [rng.uniform(0.0, 0.30) for _ in range(2)]   # never clip ink horizontally
    t = int(y0 - my[0] * h); b = int(y1 + my[1] * h)
    l = int(x0 - mx[0] * h); r = int(x1 + mx[1] * h)
    pad = max(0, -t, -l, b - mask.shape[0], r - mask.shape[1])
    if pad:
        mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        t, b, l, r = t + pad, b + pad, l + pad, r + pad
    return mask[max(0, t):b, max(0, l):r]


def add_segmentation_artifacts(mask, rng):
    """Things a real word crop often contains: ruling/underline, a sliver of the neighbour word."""
    h, w = mask.shape
    m = mask.copy()
    r = rng.random()
    if r < 0.08:   # ruled line / underline
        y = int(rng.uniform(0.75, 0.98) * h)
        cv2.line(m, (0, y), (w, y + rng.randint(-2, 2)), 255, max(1, h // 40))
    elif r < 0.14:  # neighbour fragment on left or right edge - only inside the empty margin
        cols = np.where(mask.max(axis=0) > 20)[0]
        if len(cols) == 0:
            return m
        left_gap, right_gap = int(cols[0]), int(w - 1 - cols[-1])
        side_left = rng.random() < 0.5
        gap = left_gap if side_left else right_gap
        fw = min(gap - 1, max(2, int(rng.uniform(0.02, 0.06) * w)))
        if fw < 2:
            return m
        frag = (_np_rng(rng).random((h, fw)) > 0.55).astype(np.uint8) * 255
        frag = cv2.GaussianBlur(frag, (3, 3), 0)
        frag[: int(h * 0.25)] = 0; frag[int(h * 0.8):] = 0
        if side_left:
            m[:, :fw] = np.maximum(m[:, :fw], frag)
        else:
            m[:, w - fw:] = np.maximum(m[:, w - fw:], frag)
    return m


def line_interference(mask, rng):
    """Old books use tight line spacing: tips of the lines above/below intrude into a word crop."""
    h, w = mask.shape
    m = mask.copy()
    for side in ("top", "bottom"):
        if rng.random() < 0.5:
            continue
        depth = int(h * rng.uniform(0.06, 0.18))
        if depth < 2:
            continue
        src = np.roll(mask, rng.randint(-w // 3, w // 3), axis=1)
        if side == "top":        # descenders of the line above
            m[:depth] = np.maximum(m[:depth], src[h - depth:])
        else:                    # ascenders / upper vowel signs of the line below
            m[h - depth:] = np.maximum(m[h - depth:], src[:depth])
    return m


def page_curl(mask, rng):
    """Baseline bends near the spine of a bound book (smooth vertical displacement along x)."""
    h, w = mask.shape
    amp = h * rng.uniform(0.01, 0.05)
    phase, freq = rng.uniform(0, np.pi * 2), rng.uniform(0.3, 1.2)
    xs = np.arange(w, dtype=np.float32)
    dy = amp * np.sin(phase + freq * np.pi * xs / max(1, w))
    pad = int(np.ceil(amp)) + 2
    src = cv2.copyMakeBorder(mask, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    map_x = np.tile(xs, (h + 2 * pad, 1))
    map_y = (np.arange(h + 2 * pad, dtype=np.float32)[:, None] + dy[None, :])
    return cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)


def archive_scan(img, rng):
    """How old books were digitised: bilevel scans, photocopies, microfilm."""
    kind = rng.random()
    if kind < 0.40:     # bilevel archive scan: threshold near Otsu, sometimes too high/low
        u8 = np.clip(img, 0, 255).astype(np.uint8)
        otsu, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = otsu * rng.uniform(0.85, 1.12)
        img = np.where(img > thr, 255.0, 0.0).astype(np.float32)
        if rng.random() < 0.5:
            img = cv2.medianBlur(img.astype(np.uint8), 3).astype(np.float32)
    elif kind < 0.70:   # photocopy: harsh contrast, grey cast, toner specks
        img = np.clip((img - 128) * rng.uniform(1.4, 2.2) + 128 + rng.uniform(-20, 20), 0, 255)
        npr = _np_rng(rng)
        n = int(img.size * rng.uniform(0.0005, 0.003))
        img[npr.integers(0, img.shape[0], n), npr.integers(0, img.shape[1], n)] = rng.uniform(0, 80)
    else:               # microfilm: low contrast, soft, vignetted
        img = 128 + (img - 128) * rng.uniform(0.35, 0.7)
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.8, 1.8))
        h, w = img.shape
        yy, xx = np.ogrid[:h, :w]
        v = 1 - 0.35 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        img = img * np.clip(v, 0.6, 1.0)
    return img


def legibility(img, mask):
    """Is the text still recoverable? Compares the degraded image against the clean ink mask.
    Returns (ok, stats). A human-readable crop keeps most of the ink core darker than the
    local background and does not turn much of the background dark."""
    core = cv2.erode((mask > 160).astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    if core.sum() < 10:                          # thin/small text: fall back to the un-eroded ink
        core = mask > 160
    far_bg = cv2.dilate((mask > 20).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
    if far_bg.sum() < 10:                        # tight crop: use the image border as background
        far_bg = np.zeros_like(core)
        far_bg[[0, -1], :] = True; far_bg[:, [0, -1]] = True
        far_bg &= mask < 20
    if core.sum() < 5 or far_bg.sum() < 5:
        return False, {"reason": "degenerate"}
    ink_v = float(np.median(img[core])); bg_v = float(np.median(img[far_bg]))
    contrast = bg_v - ink_v                      # negative for inverted (light-on-dark) images
    thr = (ink_v + bg_v) / 2
    if contrast >= 0:
        core_ok = float((img[core] < thr).mean()); bg_bad = float((img[far_bg] < thr).mean())
    else:
        core_ok = float((img[core] > thr).mean()); bg_bad = float((img[far_bg] > thr).mean())
    # shape agreement: an Otsu binarisation of the degraded crop should still look like the ink mask
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    flag = cv2.THRESH_BINARY_INV if contrast >= 0 else cv2.THRESH_BINARY
    _, bw = cv2.threshold(u8, 0, 1, flag + cv2.THRESH_OTSU)
    ink = mask > 100
    iou = float((bw.astype(bool) & ink).sum()) / max(1, float((bw.astype(bool) | ink).sum()))
    stats = {"contrast": abs(contrast), "core_ok": core_ok, "bg_bad": bg_bad, "iou": iou}
    # every part of the word must survive: check the ink core in vertical strips ~0.6 x height wide
    h, w = mask.shape
    strip = max(4, int(0.6 * h))
    worst = 1.0
    for x0 in range(0, w, strip):
        c = core[:, x0:x0 + strip]
        if c.sum() < 8:
            continue
        v = img[:, x0:x0 + strip][c]
        frac = float((v < thr).mean()) if contrast >= 0 else float((v > thr).mean())
        worst = min(worst, frac)
    stats.update({"iou": iou, "worst_strip": worst})
    # global agreement (catches heavy blur) and fragmentation (catches strokes broken into dust)
    a = img.astype(np.float32).ravel(); b = mask.astype(np.float32).ravel()
    corr = abs(float(np.corrcoef(a, b)[0, 1])) if a.std() > 0 and b.std() > 0 else 0.0
    nm, _, st_m, _ = cv2.connectedComponentsWithStats((mask > 100).astype(np.uint8), connectivity=8)
    areas_m = st_m[1:, cv2.CC_STAT_AREA]
    if len(areas_m):
        min_area = max(4, 0.25 * float(areas_m.min()))      # smallest real stroke = virama dot
        _, _, st_i, _ = cv2.connectedComponentsWithStats(bw.astype(np.uint8), connectivity=8)
        n_img = int((st_i[1:, cv2.CC_STAT_AREA] >= min_area).sum())
        frag = n_img / max(1, len(areas_m))
    else:
        frag = 1.0
    stats.update({"corr": corr, "fragmentation": frag})
    ok = (abs(contrast) >= 50 and core_ok >= 0.85 and bg_bad <= 0.06 and iou >= 0.40
          and worst >= 0.6 and corr >= 0.6 and frag <= 2.5)
    return ok, stats


# ------------------------------------------------------------------ compositing
def composite(mask, paper, ink_level, ink_var_field=None):
    """paper: float32 HxW background brightness 0..255; ink_level: 0..255 darkness of ink."""
    a = mask.astype(np.float32) / 255.0
    ink = np.full_like(paper, ink_level, dtype=np.float32)
    if ink_var_field is not None:
        ink = ink + ink_var_field
    return np.clip(paper * (1 - a) + ink * a, 0, 255)


def _jpeg(img, rng, qmin=25, qmax=90):
    q = rng.randint(qmin, qmax)
    ok, enc = cv2.imencode(".jpg", img.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) if ok else img


def _resolution_loss(img, rng, lo=0.35, hi=0.9):
    h, w = img.shape
    s = rng.uniform(lo, hi)
    small = cv2.resize(img, (max(4, int(w * s)), max(4, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST]))


def _illumination(img, rng, strength=40):
    h, w = img.shape
    gx = np.linspace(rng.uniform(-1, 1), rng.uniform(-1, 1), w, dtype=np.float32)
    gy = np.linspace(rng.uniform(-1, 1), rng.uniform(-1, 1), h, dtype=np.float32)
    field = (gy[:, None] + gx[None, :]) * rng.uniform(0, strength) / 2
    return img + field


# ------------------------------------------------------------------ pipelines
def modern_print(mask, rng):
    h, w = mask.shape
    npr = _np_rng(rng)
    base = rng.uniform(185, 255)
    paper = np.full((h, w), base, np.float32)
    if rng.random() < 0.5:                             # slight paper/photo texture
        paper += (_smooth_noise(h, w, rng, max(8, h)) - 0.5) * rng.uniform(0, 25)
    if rng.random() < 0.3:
        paper = _illumination(paper, rng, 60)
    ink = rng.uniform(0, 90)
    if rng.random() < 0.15:                            # inverted / dark background (signage, captions)
        paper = 255 - paper * 0.6
        ink = rng.uniform(200, 255)
    img = composite(mask, paper, ink)
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.3))
    if rng.random() < 0.35:
        img = _resolution_loss(img, rng)
    if rng.random() < 0.5:
        img = img + npr.normal(0, rng.uniform(2, 14), img.shape).astype(np.float32)
    if rng.random() < 0.10:                            # bilevel scan
        _, img = cv2.threshold(np.clip(img, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = img.astype(np.float32)
    if rng.random() < 0.5:
        img = _jpeg(np.clip(img, 0, 255), rng)
    return np.clip(img, 0, 255).astype(np.uint8)


def old_print(mask, rng):
    h, w = mask.shape
    npr = _np_rng(rng)
    m = mask.copy()
    # uneven inking: bleed (thicker) or starved (broken) strokes
    r = rng.random()
    if r < 0.4:
        k = rng.choice([2, 3])
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        m = cv2.GaussianBlur(m, (0, 0), rng.uniform(0.6, 1.4))
    elif r < 0.75:
        holes = _smooth_noise(h, w, rng, max(3, h / 6)) < rng.uniform(0.12, 0.28)
        m = np.where(holes, (m * rng.uniform(0.25, 0.6)).astype(np.uint8), m)
        if rng.random() < 0.5:
            m = cv2.erode(m, np.ones((2, 2), np.uint8))
    # aged paper: yellowed base (rendered as grey level), fibres, stains, vignetting
    base = rng.uniform(150, 225)
    paper = np.full((h, w), base, np.float32)
    paper += (_smooth_noise(h, w, rng, max(6, h * 1.5)) - 0.5) * rng.uniform(15, 55)
    fibres = npr.random((h, w)).astype(np.float32)
    paper -= (fibres > 0.985) * rng.uniform(10, 45)
    if rng.random() < 0.3:                             # stain blob
        cy, cx = rng.randint(0, h - 1), rng.randint(0, w - 1)
        rad = rng.uniform(0.3, 1.2) * h
        yy, xx = np.ogrid[:h, :w]
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2))).astype(np.float32)
        paper -= blob * rng.uniform(15, 50)
    if rng.random() < 0.35:                            # bleed-through of reverse side (mirrored faint strokes)
        ghost = cv2.flip(mask, 1)
        ghost = cv2.GaussianBlur(ghost, (0, 0), rng.uniform(1.0, 2.5)).astype(np.float32) / 255.0
        paper -= ghost * rng.uniform(15, 45)
    paper = _illumination(paper, rng, 45)
    ink_field = (_smooth_noise(h, w, rng, max(4, h / 2)) - 0.5) * rng.uniform(20, 70)
    img = composite(m, paper, rng.uniform(15, 90), ink_field)
    if rng.random() < 0.5:                             # specks / dust
        n = int(h * w * rng.uniform(0.0005, 0.004))
        ys, xs = npr.integers(0, h, n), npr.integers(0, w, n)
        img[ys, xs] = rng.uniform(20, 90)
        img = cv2.GaussianBlur(img, (3, 3), 0) if rng.random() < 0.5 else img
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.6))
    if rng.random() < 0.6:
        img = _resolution_loss(img, rng, 0.3, 0.8)
    img = img + npr.normal(0, rng.uniform(3, 16), img.shape).astype(np.float32)
    if rng.random() < 0.35:
        img = archive_scan(img, rng)
    if rng.random() < 0.6:
        img = _jpeg(np.clip(img, 0, 255), rng, 20, 80)
    return np.clip(img, 0, 255).astype(np.uint8)
