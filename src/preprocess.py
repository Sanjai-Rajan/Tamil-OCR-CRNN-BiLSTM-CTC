"""
The ONE preprocessing path used by training, evaluation and the application.
(The old project applied an unsharp mask only in the app; any such difference between
training and inference silently costs accuracy, so everything goes through here.)
"""
import cv2
import numpy as np
import torch
from PIL import Image

TARGET_H = 48          # model input height (Tamil needs room above/below for vowel signs)
MAX_W = 1024           # widest input after resizing (longer crops are squeezed horizontally)
MIN_W = 16


def to_gray(img) -> np.ndarray:
    """Accept PIL image, RGB/BGR/gray numpy array; return uint8 grayscale."""
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("L"))
    a = np.asarray(img)
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = a[:, :, :3]
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def resize_to_height(gray: np.ndarray, height: int = TARGET_H, max_w: int = MAX_W) -> np.ndarray:
    h, w = gray.shape[:2]
    if h < 2 or w < 2:
        return np.full((height, MIN_W), 255, np.uint8)
    new_w = int(round(w * height / h))
    new_w = int(np.clip(new_w, MIN_W, max_w))
    interp = cv2.INTER_AREA if h > height else cv2.INTER_CUBIC
    return cv2.resize(gray, (new_w, height), interpolation=interp)


def to_tensor(gray48: np.ndarray) -> torch.Tensor:
    """uint8 HxW -> float tensor 1xHxW in [-1, 1]."""
    t = torch.from_numpy(gray48).float().div_(255.0)
    return t.sub_(0.5).div_(0.5).unsqueeze(0)


def background_value(gray48: np.ndarray) -> float:
    """Median of the border pixels, used as padding colour so padding looks like paper."""
    border = np.concatenate([gray48[0], gray48[-1], gray48[:, 0], gray48[:, -1]])
    return float(np.median(border))


def prepare(img) -> tuple[torch.Tensor, float]:
    """Full inference-time preprocessing for one word crop -> (tensor 1xHxW, pad value in [-1,1])."""
    g = resize_to_height(to_gray(img))
    return to_tensor(g), (background_value(g) / 255.0 - 0.5) / 0.5
