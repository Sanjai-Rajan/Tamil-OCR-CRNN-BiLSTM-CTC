"""
Upload validation and decoding: one image file's bytes -> DecodedImage (2-D uint8 grayscale, as the recognizer takes it).

Order of the checks (each failure raises InputError with a message that can be shown to the user as it is):
    empty file -> file size -> protected research data, byte-identical (wordocr.guard) -> extension -> PDF -> content format (Pillow reads only the header) -> extension agrees
    with the content -> pixel dimensions -> frames -> bit depth -> full decode (corrupted or truncated data) ->
    pageocr.imageio.load_page_bytes, the project's decoder (OpenCV grayscale decode as in training; transparency put
    on white; EXIF rotation applied; images with at most two gray levels refused, as pageocr.engine does) ->
    development-page guard on the file hash and the decoded pixels (wordocr.guard, pageocr.devguard).

WEBP is not read by load_page_bytes. A WEBP upload is decoded by Pillow and re-encoded as a lossless PNG in memory
(same pixels, transparency kept) before that decoder; its EXIF rotation is applied first.

Limits are checked BEFORE any pixel is decoded (Pillow reads the header only), so an oversized image is refused
without allocating it.
"""
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import PurePath

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from pageocr import imageio as pio
from wordocr import guard
from pageocr.engine import natural_width
from src import preprocess as pp

MAX_FILE_BYTES = 20 * 1024 * 1024          # equals server.maxUploadSize in .streamlit/config.toml
MAX_PIXELS = 40_000_000                    # one word crop is far smaller; refuses decompression bombs early
MAX_SIDE = 12_000
MIN_SIDE = 2
MAX_BATCH_FILES = 50
EXTENSIONS = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "tif": "TIFF", "tiff": "TIFF", "webp": "WEBP"}
_PIL_FORMAT = {"PNG": "PNG", "JPEG": "JPEG", "MPO": "JPEG", "TIFF": "TIFF", "WEBP": "WEBP"}
SUPPORTED_TEXT = "PNG, JPG/JPEG, TIFF (single page) or WEBP"


class InputError(ValueError):
    """The upload cannot be used. str(error) is a complete, user-facing message."""


@dataclass(frozen=True)
class DecodedImage:
    name: str                  # the upload's file name, as given (display only)
    sha256: str                # of the uploaded bytes
    size_bytes: int
    format: str                # PNG, JPEG, TIFF or WEBP (the detected content, not the extension)
    mode: str                  # Pillow mode of the file (L, RGB, RGBA, P, ...)
    width: int                 # of the decoded image, after any EXIF rotation
    height: int
    exif_rotated: bool
    transparency_flattened: bool
    gray: np.ndarray = field(repr=False)      # 2-D uint8, read-only: exactly what the recognizer receives
    model_input_width: int = 0                # width at the model's height (48 px), after the 1024 px cap
    clipped: bool = False                     # True if the natural width exceeded the cap and was squeezed
    warnings: tuple = ()


def _human_size(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB" if n >= 1024 * 1024 else f"{n / 1024:.0f} KB"


def safe_stem(name: str, fallback: str = "image") -> str:
    """A file-name stem safe on every platform: ASCII letters, digits, '-' and '_' only, at most 60 characters."""
    stem = PurePath(str(name).replace("\\", "/")).name
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_-")[:60].strip("_-")
    if not stem:
        return fallback
    if stem.split("_")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}:
        stem = f"{fallback}_{stem}"
    return stem


def _gray_message(e: Exception) -> str:
    text = str(e)
    if "two gray levels" in text:
        return ("The image has at most two gray levels (pure black on pure white, binarized, or blank). The model was "
                "trained on grayscale images with soft edges; use the original photo or scan of the word, not a "
                "thresholded copy.")
    if "too large" in text:
        return f"The image is too large. Limits: {MAX_SIDE:,} px per side and {MAX_PIXELS:,} pixels; crop the word first."
    if "too small" in text:
        return "The image is too small to contain text."
    if "more than 8 bits" in text:
        return "Images with more than 8 bits per channel are not supported. Save the image as 8-bit."
    return "The image data could not be decoded. The file may be damaged; please export it again."


def decode_upload(name: str, data: bytes) -> DecodedImage:
    if not data:
        raise InputError("The file is empty (0 bytes).")
    if len(data) > MAX_FILE_BYTES:
        raise InputError(f"The file is too large ({_human_size(len(data))}). The limit is {_human_size(MAX_FILE_BYTES)}; "
                         "a cropped word image is usually well under 1 MB.")
    try:                                                     # protected research data: refused before anything else
        guard.check_bytes(data)
    except guard.ProtectedDataError as e:
        raise InputError(str(e)) from None
    ext = PurePath(str(name)).suffix.lower().lstrip(".")
    if data[:5] == b"%PDF-" or ext == "pdf":
        raise InputError("PDF files are not supported. Crop the word from the page and save it as PNG or JPG.")
    if ext not in EXTENSIONS:
        shown = f".{ext}" if re.fullmatch(r"[a-z0-9]{1,10}", ext) else ("no extension" if not ext else "unrecognized extension")
        raise InputError(f"Unsupported file type ({shown}). Supported: {SUPPORTED_TEXT}.")
    try:
        with Image.open(io.BytesIO(data)) as probe:           # reads the header only
            fmt = _PIL_FORMAT.get((probe.format or "").upper())
            width, height = probe.size
            mode, frames = probe.mode, getattr(probe, "n_frames", 1)
    except Image.DecompressionBombError:
        raise InputError(f"The image dimensions are too large. The limit is {MAX_PIXELS:,} pixels.") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise InputError(f"This file could not be read as an image; it may be damaged or not an image. Supported: {SUPPORTED_TEXT}.") from None
    if fmt is None:
        raise InputError(f"The file content is not a supported image format. Supported: {SUPPORTED_TEXT}.")
    if EXTENSIONS[ext] != fmt:
        raise InputError(f"The file name ends in .{ext} but the file is a {fmt} image. Rename it with the correct "
                         "extension and upload it again.")
    if width * height > MAX_PIXELS or max(width, height) > MAX_SIDE:
        raise InputError(f"The image is too large ({width} x {height} px). Limits: {MAX_SIDE:,} px per side and "
                         f"{MAX_PIXELS:,} pixels; crop the word first.")
    if min(width, height) < MIN_SIDE:
        raise InputError(f"The image is too small ({width} x {height} px) to contain text.")
    if frames > 1:
        raise InputError(f"The file holds {frames} images (pages or animation frames). Upload a single image.")
    if mode == "1":
        raise InputError("This is a 1-bit (black and white) image. Use the original grayscale or colour image of the word.")
    if mode.startswith(("I", "F")):
        raise InputError(f"Images with more than 8 bits per channel are not supported (mode {mode}). Save it as 8-bit.")
    try:                                                     # a full decode: catches corrupted and truncated data
        with Image.open(io.BytesIO(data)) as full:
            full.load()
            if fmt == "WEBP":
                rotated = ImageOps.exif_transpose(full)
                exif_rotated = rotated.size != full.size or full.getexif().get(0x0112, 1) not in (1, None)
                keep = rotated.convert("RGBA") if rotated.mode in ("RGBA", "LA", "PA") or "transparency" in rotated.info else rotated.convert("RGB")
                buf = io.BytesIO()
                keep.save(buf, "PNG")
                decode_bytes = buf.getvalue()
    except Exception:
        raise InputError("The image data is damaged or incomplete and could not be decoded. Please export it again.") from None
    if fmt != "WEBP":
        decode_bytes = data
    try:
        page = pio.load_page_bytes(decode_bytes, str(name))
    except pio.PageError as e:
        raise InputError(_gray_message(e)) from None
    if fmt != "WEBP":
        exif_rotated = page.exif_transposed
    with Image.open(io.BytesIO(data)) as probe:
        flattened = probe.mode in ("RGBA", "LA", "PA") or "transparency" in probe.info

    sha = hashlib.sha256(data).hexdigest()
    try:                                                     # development pages (file hash and decoded pixels)
        guard.check_image(sha, page.gray)
    except guard.ProtectedDataError as e:
        raise InputError(str(e)) from None
    h, w = page.gray.shape
    natural = natural_width(h, w)
    model_w = int(np.clip(natural, pp.MIN_W, pp.MAX_W))
    clipped = natural > pp.MAX_W
    notes = []
    if clipped:
        notes.append(f"The image is very wide for one word ({natural} px at the model's height of {pp.TARGET_H} px; "
                     f"limit {pp.MAX_W} px), so it is squeezed horizontally. This usually means more than one word: "
                     "crop a single word.")
    if h > w:
        notes.append("The image is taller than it is wide. The model reads one horizontal word; check the orientation "
                     "and the crop.")
    return DecodedImage(name=str(name), sha256=sha,
                        size_bytes=len(data), format=fmt, mode=mode, width=w, height=h, exif_rotated=bool(exif_rotated),
                        transparency_flattened=flattened, gray=page.gray, model_input_width=model_w, clipped=clipped,
                        warnings=tuple(notes))


def model_view(gray: np.ndarray) -> np.ndarray:
    """The image at the model's input height, as src/preprocess.py resizes it (for display only)."""
    return pp.resize_to_height(gray)
