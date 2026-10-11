"""
The two image paths of automatic page segmentation (V3), and the layout image.

    OCR IMAGE     the page as the recognizer must see it: the ORIGINAL grayscale pixels, optionally rotated by a small
                  deskew angle, and nothing else. Word crops for the recognizer are cut from this image only.
    LAYOUT IMAGE  a processed copy used only to FIND text: illumination-corrected, thresholded (Sauvola), cleaned of
                  specks, borders, rules and bleed-through. It is a boolean ink mask. It is never shown to the recognizer.

The rule is enforced by type, not by convention. OcrImage refuses a boolean or a two-level (binarized) array.
LayoutImage has no gray pixels to crop. pageocr.pipeline.ocr_crops accepts an OcrImage and raises TypeError for
anything else. The frozen recognizer's own input check (pageocr.engine.validate_crop) rejects two-level crops as well.

Nothing here reads an annotation or runs the recognizer. Classical image processing only (numpy, OpenCV): no model.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

LAYOUT_VERSION = "pageocr-layout-3.0.0"


class ImagePathError(TypeError):
    """An image was offered to the wrong path: a layout image to the recognizer, or a binarized image as OCR image."""


@dataclass(frozen=True)
class SegParams:
    """Every number the segmentation uses. Lengths are in units of the page's median text-component height ("h")
    unless they say otherwise, so one setting serves pages of different resolution. Recorded with every result."""
    sauvola_window_h: float = 3.0        # Sauvola window side, in h (estimated from a first pass); at least sauvola_window_min px
    sauvola_window_min: int = 25
    sauvola_k: float = 0.25
    sauvola_r: float = 128.0
    min_contrast: int = 18               # an ink pixel is at least this many gray levels darker than its local mean
    illumination_kernel_frac: float = 1 / 24   # closing kernel for the background estimate, as a fraction of the page height
    speck_area_h2: float = 0.010         # components smaller than this x h^2 ...
    speck_side_h: float = 0.12           # ... and no larger than this x h on either side are specks
    rule_aspect: float = 18.0            # a component this elongated ...
    rule_length_h: float = 6.0           # ... and at least this long is a rule line, not text
    border_reach_frac: float = 0.45      # a component touching the page edge and this long (fraction of the page side) is a border
    bleed_lightness: float = 0.58        # components whose mean darkness is below this share of the page's ink darkness are bleed-through
    huge_component_h: float = 6.0        # components taller than this x h are figures or ornaments, not text
    deskew_max_deg: float = 5.0
    deskew_min_deg: float = 0.2          # smaller measured angles are left alone
    gutter_min_h: float = 1.0            # a column gutter is at least this wide ...
    gutter_min_lines: int = 4            # ... and is blank across at least this many consecutive text strips
    line_split_factor: float = 1.75      # a text band this many times the median band height is split (touching lines)
    line_thin_factor: float = 0.45       # a band thinner than this x the median band height is a row of marks, joined to a neighbour
    mark_side_h: float = 0.42            # a component no larger than this x line height is a mark (pulli, dot, superscript, comma)
    mark_attach_h: float = 0.18          # a mark this close (x line height) to a letter group joins it
    gap_floor_h: float = 0.16            # no word gap is narrower than this x line height
    gap_default_h: float = 0.34          # the word-gap threshold of a line with too few gaps to decide for itself
    gap_separation: float = 1.6          # a line's own threshold is trusted if its wide gaps are this many times its narrow ones
    punctuation_attach: float = 1.25     # trailing punctuation joins the word before it if the gap is below this x the line's threshold
    word_pad_px: int = 0                 # word boxes are tight ink boxes; padding belongs to crop time
    # EXPERIMENTAL, opt-in, off by default (diagnostic workspace 2026-10-10). Off = the behaviour above, unchanged.
    exp_rule_removal: bool = False       # remove long horizontal rule pixels from the ink mask before components (pageocr.rules)
    exp_contrast_clahe: bool = False     # local contrast equalisation (CLAHE) of the illumination-corrected image before thresholding
    exp_angle_lines: bool = False        # find lines by following the letters along their own direction (pageocr.slant_lines)


@dataclass(frozen=True)
class OcrImage:
    """The image word crops are cut from: original grayscale, optionally deskewed. See the module docstring."""
    gray: np.ndarray                     # 2-D uint8, read-only
    page_sha256: str = ""
    deskew_deg: float = 0.0              # rotation applied to the original, 0.0 if none
    to_original: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))    # 2 x 3 affine: a point of this image -> the original page
    original_shape: tuple = ()           # (height, width) of the original page

    def __post_init__(self):
        g = self.gray
        if not isinstance(g, np.ndarray):
            raise ImagePathError(f"an OCR image is a numpy array, got {type(g).__name__}")
        if g.dtype == np.bool_:
            raise ImagePathError("a boolean mask is a layout image; the recognizer is given original grayscale only")
        if g.ndim != 2 or g.dtype != np.uint8:
            raise ImagePathError(f"an OCR image is 2-D uint8 grayscale, got shape {g.shape} dtype {g.dtype}")
        if np.count_nonzero(np.bincount(g.ravel(), minlength=256)) <= 2:
            raise ImagePathError("this image has at most two gray levels: it is binarized. The recognizer is given original grayscale only")
        if g.flags.writeable:
            g = g.copy()
            g.setflags(write=False)
            object.__setattr__(self, "gray", g)
        if not self.original_shape:
            object.__setattr__(self, "original_shape", tuple(int(v) for v in g.shape))

    @property
    def shape(self) -> tuple:
        return self.gray.shape

    @staticmethod
    def from_page(page) -> "OcrImage":
        """From a pageocr.imageio.Page (the original page, decoded once)."""
        return OcrImage(gray=page.gray, page_sha256=page.sha256)

    def rotated(self, angle_deg: float) -> "OcrImage":
        """This image rotated about its centre by angle_deg (counter-clockwise), on a canvas large enough to keep every
        pixel. Bilinear resampling of the gray values; the border is filled with the median of the page's edge pixels."""
        if self.deskew_deg != 0.0:
            raise ImagePathError("this OCR image is already deskewed; rotate the original once")
        h, w = self.gray.shape
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(np.ceil(w * cos + h * sin)), int(np.ceil(w * sin + h * cos))
        m[0, 2] += nw / 2.0 - w / 2.0
        m[1, 2] += nh / 2.0 - h / 2.0
        edge = np.concatenate([self.gray[0], self.gray[-1], self.gray[:, 0], self.gray[:, -1]])
        out = cv2.warpAffine(self.gray, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float(np.median(edge)))
        inv = cv2.invertAffineTransform(m)
        return OcrImage(gray=out, page_sha256=self.page_sha256, deskew_deg=float(angle_deg),
                        to_original=tuple(tuple(float(v) for v in row) for row in inv), original_shape=(h, w))

    def box_in_original(self, box) -> tuple:
        """The axis-aligned box, in ORIGINAL page pixels, that holds a box (x1, y1, x2, y2) of this image."""
        x1, y1, x2, y2 = box
        if self.deskew_deg == 0.0:
            return (int(x1), int(y1), int(x2), int(y2))
        a = np.array(self.to_original)
        pts = np.array([[x1, y1, 1.0], [x2, y1, 1.0], [x1, y2, 1.0], [x2, y2, 1.0]]) @ a.T
        H, W = self.original_shape
        ox1, oy1 = int(max(0, np.floor(pts[:, 0].min()))), int(max(0, np.floor(pts[:, 1].min())))
        ox2, oy2 = int(min(W, np.ceil(pts[:, 0].max()))), int(min(H, np.ceil(pts[:, 1].max())))
        return (ox1, oy1, max(ox1 + 1, ox2), max(oy1 + 1, oy2))


@dataclass(frozen=True)
class LayoutImage:
    """What the segmentation looks at. A boolean ink mask and its connected components; no gray values."""
    ink: np.ndarray                      # 2-D bool, True where text ink was found
    components: np.ndarray               # N x 5 int: x1, y1, x2, y2 (exclusive), area - text components only
    h: float                             # median height of the text components: the unit of length of this page
    removed: dict = field(default_factory=dict)       # how many components each cleaning step took out
    steps: tuple = ()                    # the processing steps applied, in order
    flags: tuple = ()

    def __post_init__(self):
        if not isinstance(self.ink, np.ndarray) or self.ink.dtype != np.bool_ or self.ink.ndim != 2:
            raise ImagePathError("a layout image is a 2-D boolean ink mask")


def _odd(n: float, lo: int = 3) -> int:
    n = max(lo, int(round(n)))
    return n if n % 2 else n + 1


def correct_illumination(gray: np.ndarray, kernel_frac: float) -> np.ndarray:
    """Divide the page by an estimate of its background (large morphological closing, blurred). uint8 out, paper near 255."""
    h, w = gray.shape
    scale = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(gray, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else gray
    k = _odd(max(small.shape) * kernel_frac, 9)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (_odd(k * 1.5), _odd(k * 1.5)), 0)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR) if scale < 1 else bg
    return np.clip(gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0) * 255.0, 0, 255).astype(np.uint8)


def sauvola(norm: np.ndarray, window: int, k: float, r: float, min_contrast: int) -> np.ndarray:
    """Sauvola's local threshold, with a floor on local contrast so flat paper yields no ink. Boolean out."""
    f = norm.astype(np.float32)
    mean = cv2.boxFilter(f, cv2.CV_32F, (window, window), borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(f * f, cv2.CV_32F, (window, window), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    return (f < mean * (1.0 + k * (std / r - 1.0))) & (f < mean - min_contrast)


def _components(ink: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    s = stats[1:]
    boxes = np.stack([s[:, 0], s[:, 1], s[:, 0] + s[:, 2], s[:, 1] + s[:, 3], s[:, 4]], axis=1).astype(np.int64) if n > 1 else np.zeros((0, 5), np.int64)
    return labels, boxes


def text_height(boxes: np.ndarray) -> float:
    """Median height of the components large enough to be letters."""
    if len(boxes) == 0:
        return 0.0
    hgt = (boxes[:, 3] - boxes[:, 1]).astype(float)
    big = hgt[(boxes[:, 4] >= 12) & (hgt >= 6)]
    if len(big) < 5:
        return float(np.median(hgt)) if len(hgt) else 0.0
    upper = big[big >= np.percentile(big, 40)]            # marks and dots pull the plain median down
    return float(np.median(upper))


def build_layout_image(ocr: OcrImage, params: SegParams = SegParams()) -> LayoutImage:
    """The layout image of an OCR image. The OCR image is read, never changed."""
    if not isinstance(ocr, OcrImage):
        raise ImagePathError("build_layout_image takes an OcrImage")
    gray = ocr.gray
    H, W = gray.shape
    steps, removed, flags = [], {}, []
    norm = correct_illumination(gray, params.illumination_kernel_frac)
    steps.append("illumination correction (division by a closing-based background estimate)")
    if params.exp_contrast_clahe:
        norm = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(norm)
        steps.append("EXPERIMENTAL: local contrast equalisation (CLAHE, clip 2.0, 8 x 8 tiles)")
    # first pass with the minimum window, to learn the size of the print; second pass with a window matched to it
    ink = sauvola(norm, _odd(params.sauvola_window_min), params.sauvola_k, params.sauvola_r, params.min_contrast)
    _, boxes = _components(ink)
    h0 = text_height(boxes)
    window = _odd(max(params.sauvola_window_min, params.sauvola_window_h * h0))
    if window != _odd(params.sauvola_window_min):
        ink = sauvola(norm, window, params.sauvola_k, params.sauvola_r, params.min_contrast)
    steps.append(f"Sauvola threshold (window {window} px, k {params.sauvola_k}, contrast floor {params.min_contrast})")
    if params.exp_rule_removal and h0:
        from pageocr.rules import remove_rules
        ink, rule_mask = remove_rules(ink, float(h0))
        removed["rule pixels (experimental)"] = int(rule_mask.sum())
        steps.append(f"EXPERIMENTAL: rule-line pixels removed from the ink mask ({int(rule_mask.any(axis=1).sum())} rows touched)")
    labels, boxes = _components(ink)
    if len(boxes) == 0:
        return LayoutImage(ink=np.zeros((H, W), bool), components=boxes, h=0.0, removed={}, steps=tuple(steps), flags=("no ink found",))
    h = text_height(boxes) or 1.0
    bw, bh, ar = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1], boxes[:, 4]
    keep = np.ones(len(boxes), bool)

    def drop(mask, name):
        nonlocal keep
        m = mask & keep
        removed[name] = int(m.sum())
        keep &= ~mask

    touches = (boxes[:, 0] <= 1) | (boxes[:, 1] <= 1) | (boxes[:, 2] >= W - 1) | (boxes[:, 3] >= H - 1)
    drop(touches & ((bw >= params.border_reach_frac * W) | (bh >= params.border_reach_frac * H)), "page border or scan edge")
    long_side, short_side = np.maximum(bw, bh), np.maximum(1, np.minimum(bw, bh))
    drop((long_side / short_side >= params.rule_aspect) & (long_side >= params.rule_length_h * h), "rule line")
    drop((bh >= params.huge_component_h * h) | (bw >= 0.6 * W), "figure, ornament or frame")
    # bleed-through and stains: components much lighter than the page's own ink
    sums = np.bincount(labels.ravel(), weights=(255.0 - norm.ravel()), minlength=len(boxes) + 1)[1:]
    dark = sums / np.maximum(ar, 1)
    letters = keep & (bh >= 0.5 * h)
    if letters.sum() >= 5:
        ref = float(np.median(dark[letters]))
        drop(dark < params.bleed_lightness * ref, "bleed-through or stain (too light)")
    drop((ar < params.speck_area_h2 * h * h) & (np.maximum(bw, bh) <= params.speck_side_h * h), "speck")
    steps.append("component cleaning: border, rule, figure, bleed-through, speck (sizes relative to the print height)")
    kept_ids = np.nonzero(keep)[0] + 1
    lut = np.zeros(len(boxes) + 1, bool)
    lut[kept_ids] = True
    clean = lut[labels]
    if removed.get("rule line", 0) >= 6:
        flags.append("many rule lines: table or form suspected")
    if removed.get("figure, ornament or frame", 0):
        flags.append("large non-text component removed: figure, ornament or frame")
    return LayoutImage(ink=clean, components=boxes[keep], h=float(text_height(boxes[keep]) or h), removed=removed, steps=tuple(steps), flags=tuple(flags))
