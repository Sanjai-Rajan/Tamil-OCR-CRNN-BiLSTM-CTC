"""
Skew of a page, measured on the layout image.

The angle is the rotation that makes the text lines horizontal: the one that maximises the sharpness of the row
profile of the ink (sum of squared differences between neighbouring rows). Searched in two passes inside +/- the
limit. Only the measurement happens here; the OCR image is rotated by OcrImage.rotated, once, and the layout image is
then rebuilt from the rotated OCR image, so the two paths stay in the same frame.
"""
import cv2
import numpy as np


def _score(mask_u8: np.ndarray, angle: float) -> float:
    h, w = mask_u8.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rot = cv2.warpAffine(mask_u8, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rows = rot.sum(axis=1, dtype=np.float64)
    d = np.diff(rows)
    return float((d * d).sum())


def estimate_skew(ink: np.ndarray, max_deg: float = 5.0) -> dict:
    """{"angle_deg": rotation (counter-clockwise, degrees) that levels the lines, "sharpness_gain": score ratio vs 0}."""
    if ink.dtype != np.bool_ or ink.ndim != 2:
        raise TypeError("estimate_skew takes the boolean ink mask of a layout image")
    if not ink.any():
        return {"angle_deg": 0.0, "sharpness_gain": 1.0}
    h, w = ink.shape
    scale = min(1.0, 1100.0 / max(h, w))
    # The mask is scaled to 0/255 before the area-average, so a thin stroke that covers part of a reduced pixel keeps
    # it (a 0/1 mask would average to below 0.5 and round to 0, and thin print would vanish before measurement).
    mask = ink.astype(np.uint8) * 255
    small = cv2.resize(mask, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else mask
    small = (small > 0).astype(np.uint8)
    coarse = np.arange(-max_deg, max_deg + 1e-9, 0.5)
    best = max(coarse, key=lambda a: (_score(small, float(a)), -abs(a)))
    fine = np.arange(best - 0.5, best + 0.5 + 1e-9, 0.1)
    best = max(fine, key=lambda a: (_score(small, float(a)), -abs(a)))
    best = float(np.clip(round(float(best), 2), -max_deg, max_deg))
    zero = _score(small, 0.0)
    return {"angle_deg": best, "sharpness_gain": float(_score(small, best) / zero) if zero > 0 else 1.0}
