"""
Page loading: one decode to a 2-D uint8 grayscale array, with the file's SHA-256, and the preview derived from it.

    page = load_page("page.png")              # or load_page_bytes(data, name) for an upload
    page.gray, page.sha256, page.shape, page.dtype, page.path

PNG, JPG/JPEG and TIFF (single page, 8 bits per channel). No PDF. The array returned is the ORIGINAL page for
everything downstream: boxes are in its pixel coordinates, crops are cut from it, and the on-screen preview is a
reduced copy of this same array (make_preview), never a second decode.

Decoding follows the project's existing reader (src/data/datasets.read_image and app.py): OpenCV's grayscale decode,
with Pillow used when the file has transparency (put on white paper) or an EXIF rotation (applied, so the array is
the page as a viewer shows it). No binarization, contrast change, denoising or deskew happens here.

Rejected: files that are not one of the formats above, PDFs, multi-page TIFFs, 1-bit and 16-bit images, pages with
only two gray levels (binarized scans: the recognizer was trained on grayscale), pages larger than MAX_PIXELS or
MAX_SIDE (checked from the header, before decoding), and any path inside protected research data.
"""
import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from pageocr import ROOT, assert_path_allowed, protected
from pageocr.boxes import PreviewScale

FORMATS = {"PNG": "PNG", "JPEG": "JPEG", "MPO": "JPEG", "TIFF": "TIFF"}       # Pillow format name -> reported name
EXTENSIONS = ("png", "jpg", "jpeg", "tif", "tiff")
# Folders indexed on this computer (pageocr.protected): the project's own real_docs/ and whatever is named in the
# environment variable TAMIL_OCR_PROTECTED_ROOTS. No folder is located through the user's home folder any more; the
# protected files themselves are recognised by the fingerprints in pageocr/protected_manifest.json wherever this runs.
PROTECTED_ROOTS = (ROOT / "real_docs",)
DEFAULT_PREVIEW_WIDTH = 1100
# Largest page accepted, checked from the file header before decoding (a 600 dpi A4 scan is about 35 megapixels;
# the largest page in this project is about 9). Pillow itself refuses only above about 179 megapixels.
MAX_PIXELS = 50_000_000
MAX_SIDE = 20_000


class PageError(ValueError):
    """The page image cannot be used; the message can be shown to the operator as it is."""


@dataclass(frozen=True)
class Page:
    gray: np.ndarray          # 2-D uint8, read-only: the original page
    path: str                 # resolved path, or the upload's file name
    sha256: str               # of the file's bytes
    shape: tuple              # (height, width)
    dtype: str                # "uint8"
    format: str               # "PNG", "JPEG" or "TIFF"
    exif_transposed: bool     # True if an EXIF rotation was applied while decoding

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[0]


def _bits_per_sample(data: bytes, fmt: str) -> int:
    """Bit depth per channel declared in a PNG's IHDR or a TIFF's BitsPerSample tag; 8 if not stated or unreadable."""
    if fmt == "PNG" and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return data[24]
    if fmt == "TIFF":
        try:
            with Image.open(io.BytesIO(data)) as im:
                bits = im.tag_v2.get(258, 8)
            return max(bits) if isinstance(bits, (tuple, list)) else int(bits)
        except Exception:
            return 8
    return 8


def load_page(path) -> Page:
    p = assert_path_allowed(path)
    with open(p, "rb") as f:
        data = f.read()
    return load_page_bytes(data, str(p))


def load_page_bytes(data: bytes, name: str = "") -> Page:
    """Decode an image file's bytes once. `name` is only recorded (a path or an upload's file name)."""
    if data[:5] == b"%PDF-":
        raise PageError("PDF files are not supported in V2. Export the page as PNG, JPG or TIFF.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = (probe.format or "").upper()
            probe.verify()
        pil = Image.open(io.BytesIO(data))
        mode, frames = pil.mode, getattr(pil, "n_frames", 1)
        orientation = pil.getexif().get(0x0112, 1)
        has_alpha = mode in ("RGBA", "LA", "PA") or "transparency" in pil.info
        width, height = pil.size                           # from the header: nothing has been decoded yet
    except Image.DecompressionBombError:
        raise PageError(f"The image is too large. The limit is {MAX_PIXELS // 1_000_000} megapixels "
                        f"and {MAX_SIDE:,} px per side.") from None
    except Exception:
        raise PageError("This file could not be read as an image. Supported: PNG, JPG/JPEG, TIFF.") from None
    if width * height > MAX_PIXELS or max(width, height) > MAX_SIDE:
        # A small compressed file can declare a huge page (a "decompression bomb"); refuse it before any decode.
        raise PageError(f"The image is too large ({width:,} x {height:,} px). The limit is {MAX_PIXELS // 1_000_000} "
                        f"megapixels and {MAX_SIDE:,} px per side.")
    if fmt not in FORMATS:
        raise PageError(f"Unsupported image format ({fmt or 'unknown'}). Supported: PNG, JPG/JPEG, TIFF.")
    if fmt == "TIFF" and frames > 1:
        raise PageError(f"This TIFF has {frames} pages. Save the page you need as a single-page image.")
    if mode == "1":
        raise PageError("This is a 1-bit (black and white) image. V2 needs the original grayscale or colour scan.")
    bits = _bits_per_sample(data, fmt)       # 16-bit colour PNG/TIFF opens in Pillow as 8-bit "RGB" (silently reduced)
    if mode.startswith(("I", "F")) or bits > 8:
        raise PageError(f"Images with more than 8 bits per channel are not supported (mode {mode}, {bits} bits per channel). "
                        "Convert to 8-bit.")

    transposed = orientation not in (1, None)
    gray = None
    try:
        if not (has_alpha or transposed or fmt == "MPO"):
            gray = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)     # as datasets.read_image
        if gray is None:
            pil = ImageOps.exif_transpose(pil)
            if has_alpha:                                  # transparent pixels are put on white paper
                rgba = pil.convert("RGBA")
                flat = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                flat.alpha_composite(rgba)
                pil = flat
            gray = np.asarray(pil.convert("L"))
    except Exception:
        raise PageError("The image data could not be decoded.") from None
    gray = np.ascontiguousarray(gray)
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise PageError("The image did not decode to 8-bit grayscale.")
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        raise PageError("The image is too small to contain text.")
    if np.count_nonzero(np.bincount(gray.ravel(), minlength=256)) <= 2:
        raise PageError("This page has at most two gray levels (a binarized or blank image). "
                        "V2 needs the original grayscale or colour scan.")
    gray.setflags(write=False)
    return Page(gray=gray, path=name, sha256=hashlib.sha256(data).hexdigest(), shape=tuple(gray.shape),
                dtype=str(gray.dtype), format=FORMATS[fmt], exif_transposed=transposed)


# ---------------------------------------------------------------- protected content (for uploads, which have no path)

def protected_size_index(roots=None) -> dict[int, list[str]]:
    """File size -> paths, for every file under the protected folders. Reads directory entries only.

    Without an argument: PROTECTED_ROOTS and the folders configured in TAMIL_OCR_PROTECTED_ROOTS; a configured folder
    that does not exist raises pageocr.protected.GuardUnavailable (fail closed), it is not skipped."""
    if roots is None:
        roots = tuple(PROTECTED_ROOTS) + protected.configured_roots()
    index: dict[int, list[str]] = {}
    for root in roots:
        for folder, _, files in os.walk(root):
            for name in files:
                p = os.path.join(folder, name)
                try:
                    index.setdefault(os.path.getsize(p), []).append(p)
                except OSError:
                    pass
    return index


def protected_match(data: bytes, index: dict[int, list[str]]) -> str | None:
    """Path of a protected file with exactly these bytes, or None. Only files of the same size are opened (read-only).

    The fingerprints in pageocr/protected_manifest.json are asked first (the result is then protected.MANIFEST_HIT);
    if they cannot be read, pageocr.protected.GuardUnavailable is raised: the caller must refuse the input."""
    if protected.manifest_match(data):
        return protected.MANIFEST_HIT
    candidates = index.get(len(data), [])
    if not candidates:
        return None
    digest = hashlib.sha256(data).digest()
    for p in candidates:
        try:
            with open(p, "rb") as f:
                if hashlib.sha256(f.read()).digest() == digest:
                    return p
        except OSError:
            pass
    return None


# ---------------------------------------------------------------- preview

def make_preview(page_gray: np.ndarray, max_width: int = DEFAULT_PREVIEW_WIDTH) -> tuple[np.ndarray, PreviewScale]:
    """A display copy of the page, never wider than max_width and never enlarged, and its mapping to the original.

    The preview is for the screen only. Boxes stay in original coordinates and crops are cut from the original.
    """
    if not isinstance(page_gray, np.ndarray) or page_gray.ndim != 2 or page_gray.dtype != np.uint8:
        raise PageError("make_preview needs the original page as a 2-D uint8 grayscale array")
    h, w = page_gray.shape
    pw = min(w, int(max_width))
    ph = max(1, int(round(h * pw / w)))
    preview = page_gray.copy() if pw == w else cv2.resize(page_gray, (pw, ph), interpolation=cv2.INTER_AREA)
    return preview, PreviewScale(original_width=w, original_height=h, preview_width=pw, preview_height=ph)


def draw_boxes(preview: np.ndarray, boxes, scale: PreviewScale) -> np.ndarray:
    """An RGB copy of the preview with each box outlined and labelled with its id (display only)."""
    if preview.shape != (scale.preview_height, scale.preview_width):
        raise PageError("the preview does not match the scale record")
    img = Image.fromarray(preview).convert("RGB")
    draw = ImageDraw.Draw(img)
    for b in boxes:
        x0, y0, x1, y1 = scale.rect_to_preview(b)
        draw.rectangle([x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)], outline=(214, 39, 40), width=2)
        label = b.id.encode("ascii", "replace").decode()
        tx, ty = x0 + 2, max(0, y0 - 12)
        tw = draw.textlength(label)
        draw.rectangle([tx - 2, ty, tx + tw + 2, ty + 11], fill=(255, 255, 255))
        draw.text((tx, ty), label, fill=(23, 60, 160))
    return np.asarray(img)
