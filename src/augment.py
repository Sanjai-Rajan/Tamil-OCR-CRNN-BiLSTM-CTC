"""
Augmentation for REAL word images (handwritten and printed crops).
Synthetic images already carry their own variation, so they skip this.

The old model trained with no augmentation and memorised its training set
(train loss 0.2 vs validation loss 1.8). These transforms target the variation
real documents have: pen width, slant, scale, blur, lighting, compression, crop margins.
"""
import random

import cv2
import numpy as np


def _affine(img, rng, max_rot=3.0, max_shear=0.35, scale_x=(0.8, 1.2)):
    h, w = img.shape
    bg = int(np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])))
    sx = rng.uniform(*scale_x)
    shear = rng.uniform(-max_shear, max_shear)
    rot = np.deg2rad(rng.uniform(-max_rot, max_rot))
    c, s = np.cos(rot), np.sin(rot)
    A = np.array([[sx * c, -s + shear, 0], [sx * s, c, 0]], np.float32)
    # compute output size that keeps the whole word
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ A.T
    x0, y0 = corners.min(0); x1, y1 = corners.max(0)
    A[:, 2] -= [x0, y0]
    W, H = int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0))
    return cv2.warpAffine(img, A, (max(4, W), max(4, H)), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


def _grid_distort(img, rng, strength=0.03):
    """Smooth elastic distortion - varies letter shapes like different handwriting."""
    h, w = img.shape
    gh, gw = 3, max(3, w // max(1, h))
    dx = cv2.resize(np.float32(np.random.default_rng(rng.getrandbits(32)).normal(0, 1, (gh, gw))), (w, h))
    dy = cv2.resize(np.float32(np.random.default_rng(rng.getrandbits(32)).normal(0, 1, (gh, gw))), (w, h))
    amp = strength * h
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, xs + dx * amp, ys + dy * amp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _stroke_width(img, rng):
    k = np.ones((2, 2), np.uint8)
    # dark ink on light paper: erode image = thicker ink, dilate = thinner ink
    return cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k)


def _photometric(img, rng):
    f = img.astype(np.float32)
    if rng.random() < 0.5:                                  # contrast / brightness
        f = (f - 128) * rng.uniform(0.6, 1.3) + 128 + rng.uniform(-30, 30)
    if rng.random() < 0.3:                                  # gamma
        f = 255 * (np.clip(f, 0, 255) / 255) ** rng.uniform(0.7, 1.5)
    if rng.random() < 0.3:                                  # lighting gradient
        h, w = f.shape
        g = np.linspace(rng.uniform(-30, 30), rng.uniform(-30, 30), w, dtype=np.float32)
        f += g[None, :]
    if rng.random() < 0.35:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.3:
        f += np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(2, 10), f.shape)
    f = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 90)])
        if ok:
            f = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return f


def _margins(img, rng):
    """Random extra margin or a tight crop of the existing margin (never cuts ink)."""
    h, w = img.shape
    bg = int(np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])))
    ink = img < (bg - 40)
    ys, xs = np.where(ink)
    if len(xs) and rng.random() < 0.4:                      # tighten towards the ink
        pad = int(0.08 * h)
        y0 = max(0, ys.min() - rng.randint(0, pad)); y1 = min(h, ys.max() + 1 + rng.randint(0, pad))
        x0 = max(0, xs.min() - rng.randint(0, pad)); x1 = min(w, xs.max() + 1 + rng.randint(0, pad))
        img = img[y0:y1, x0:x1]
    elif rng.random() < 0.4:                                # add paper around it
        t, b, l, r = (rng.randint(0, int(0.2 * h) + 1) for _ in range(4))
        img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=bg)
    return img


def _frame_lines(img, rng):
    """Leftover form/box lines at the crop edges (seen in the IIIT word crops): teach the model to ignore them."""
    h, w = img.shape
    img = img.copy()
    bg = int(np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])))
    ink = max(0, bg - rng.randint(120, 220))
    t = rng.randint(1, max(1, h // 30) + 1)
    for side in rng.sample(["top", "bottom", "left", "right"], rng.randint(1, 2)):
        off = rng.randint(0, max(1, h // 12))
        if side == "top":
            img[off:off + t, :] = ink
        elif side == "bottom":
            img[h - off - t:h - off, :] = ink
        elif side == "left":
            img[:, off:off + t] = ink
        else:
            img[:, w - off - t:w - off] = ink
    return img


class RealAugment:
    """kind='hw' for handwriting (adds elastic distortion, stronger slant); 'print' for printed crops."""

    def __init__(self, kind="hw", p=0.85):
        self.kind, self.p = kind, p

    def __call__(self, img: np.ndarray, rng: random.Random) -> np.ndarray:
        if rng.random() > self.p:
            return img
        # work at a moderate resolution: these datasets store ~300 px tall crops
        h, w = img.shape
        if h > 96:
            img = cv2.resize(img, (max(8, int(w * 96 / h)), 96), interpolation=cv2.INTER_AREA)
        img = _margins(img, rng)
        if rng.random() < 0.6:
            img = _affine(img, rng, max_shear=0.35 if self.kind == "hw" else 0.15,
                          max_rot=3.0 if self.kind == "hw" else 1.5)
        if self.kind == "hw" and rng.random() < 0.4:
            img = _grid_distort(img, rng)
        if rng.random() < 0.25:
            img = _stroke_width(img, rng)
        if rng.random() < 0.15:
            img = _frame_lines(img, rng)
        return _photometric(img, rng)
