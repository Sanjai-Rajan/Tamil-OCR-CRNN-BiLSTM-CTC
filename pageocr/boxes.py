"""
Word boxes: the schema, JSON and CSV files, validation, cropping, and preview <-> original coordinate mapping.

All box coordinates are integers in ORIGINAL page pixels: x, y is the top-left corner, w, h the size, so the box
covers columns x .. x+w-1 and rows y .. y+h-1 (x1 = x+w and y1 = y+h are exclusive). Preview coordinates exist only
on screen and are converted with PreviewScale; a crop is always cut from the original page, never from a preview.

Reading order is the operator's: column_index, then line_index, then the order of the boxes in the file. Nothing in
this package derives an order from the image or from the recognizer.

JSON file
    {"schema_version": 1,
     "page": {"width": 2480, "height": 3508, "sha256": "..."},          (optional)
     "boxes": [{"id": "w001", "x": 10, "y": 20, "w": 120, "h": 48, "line_index": 0, "column_index": 0}, ...]}
CSV file (UTF-8, one row per box, schema_version repeated in every row)
    schema_version,id,x,y,w,h,line_index,column_index
"""
import csv
import io
import json
import re
from dataclasses import asdict, dataclass

import numpy as np

from pageocr import assert_path_allowed

SCHEMA_VERSION = 1
FIELDS = ("id", "x", "y", "w", "h", "line_index", "column_index")
INT_FIELDS = FIELDS[1:]
CSV_HEADER = ("schema_version",) + FIELDS
_INT_TEXT = re.compile(r"-?[0-9]+")        # ASCII digits only: \d would also accept Tamil, Arabic-Indic, Devanagari ... digits


class BoxError(ValueError):
    """A box, a box file or a page/box combination is invalid."""


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


@dataclass(frozen=True)
class Box:
    id: str
    x: int
    y: int
    w: int
    h: int
    line_index: int
    column_index: int

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip() or self.id != self.id.strip():
            raise BoxError(f"box id must be a non-empty string without surrounding spaces, got {self.id!r}")
        for name in INT_FIELDS:
            v = getattr(self, name)
            if not _is_int(v):
                raise BoxError(f"box {self.id}: {name} must be an integer, got {v!r}")
            object.__setattr__(self, name, int(v))
        if self.x < 0 or self.y < 0:
            raise BoxError(f"box {self.id}: x and y must not be negative (x={self.x}, y={self.y})")
        if self.w <= 0 or self.h <= 0:
            raise BoxError(f"box {self.id}: w and h must be greater than 0 (w={self.w}, h={self.h})")
        if self.line_index < 0 or self.column_index < 0:
            raise BoxError(f"box {self.id}: line_index and column_index must not be negative")

    @property
    def x1(self) -> int:      # exclusive
        return self.x + self.w

    @property
    def y1(self) -> int:      # exclusive
        return self.y + self.h


@dataclass(frozen=True)
class BoxFile:
    schema_version: int
    boxes: tuple
    page: dict | None = None      # {"width", "height", "sha256"} of the page the boxes were made for, if recorded


def validate_boxes(boxes, page_width: int, page_height: int) -> None:
    """Raise BoxError unless every box is a valid Box, lies inside the page, and the ids are unique."""
    if not _is_int(page_width) or not _is_int(page_height) or page_width <= 0 or page_height <= 0:
        raise BoxError(f"invalid page size {page_width!r} x {page_height!r}")
    seen = set()
    for b in boxes:
        if not isinstance(b, Box):
            raise BoxError(f"not a Box: {b!r}")
        if b.id in seen:
            raise BoxError(f"duplicate box id {b.id!r}")
        seen.add(b.id)
        if b.x1 > page_width or b.y1 > page_height:
            raise BoxError(f"box {b.id} (x={b.x}, y={b.y}, w={b.w}, h={b.h}) is outside the page "
                           f"({page_width} x {page_height})")


def crop_from_page(page_gray: np.ndarray, box: Box) -> np.ndarray:
    """The box cut from the ORIGINAL grayscale page: page_gray[y:y+h, x:x+w], a uint8 copy, no resampling."""
    if not isinstance(page_gray, np.ndarray) or page_gray.ndim != 2 or page_gray.dtype != np.uint8:
        raise BoxError("crop_from_page needs the original page as a 2-D uint8 grayscale array")
    validate_boxes([box], int(page_gray.shape[1]), int(page_gray.shape[0]))
    return page_gray[box.y:box.y1, box.x:box.x1].copy()


def reading_order(boxes) -> list[int]:
    """Indices of `boxes` by (column_index, line_index), keeping the supplied order inside a line."""
    return sorted(range(len(boxes)), key=lambda i: (boxes[i].column_index, boxes[i].line_index))


# ---------------------------------------------------------------- preview <-> original

@dataclass(frozen=True)
class PreviewScale:
    """The explicit mapping between a displayed preview and the original page. Scale factors are original / preview."""
    original_width: int
    original_height: int
    preview_width: int
    preview_height: int

    def __post_init__(self):
        for name in ("original_width", "original_height", "preview_width", "preview_height"):
            v = getattr(self, name)
            if not _is_int(v) or v <= 0:
                raise BoxError(f"{name} must be a positive integer, got {v!r}")

    @property
    def scale_x(self) -> float:
        return self.original_width / self.preview_width

    @property
    def scale_y(self) -> float:
        return self.original_height / self.preview_height

    def to_original(self, px: float, py: float) -> tuple[int, int]:
        """A preview point as the nearest original pixel edge, clamped to the page."""
        x = int(round(px * self.original_width / self.preview_width))
        y = int(round(py * self.original_height / self.preview_height))
        return min(max(x, 0), self.original_width), min(max(y, 0), self.original_height)

    def to_preview(self, x: float, y: float) -> tuple[float, float]:
        return x * self.preview_width / self.original_width, y * self.preview_height / self.original_height

    def rect_to_preview(self, box: Box) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) of the box in preview coordinates, for drawing only."""
        return self.to_preview(box.x, box.y) + self.to_preview(box.x1, box.y1)

    def rect_to_original(self, px0: float, py0: float, px1: float, py1: float) -> tuple[int, int, int, int]:
        """A rectangle given by two preview corners as (x, y, w, h) in original pixels. Raises if it is empty."""
        x0, y0 = self.to_original(min(px0, px1), min(py0, py1))
        x1, y1 = self.to_original(max(px0, px1), max(py0, py1))
        if x1 <= x0 or y1 <= y0:
            raise BoxError("the rectangle is empty in original page pixels")
        return x0, y0, x1 - x0, y1 - y0


# ---------------------------------------------------------------- JSON

def _check_version(v, where: str) -> int:
    if not _is_int(v) or int(v) != SCHEMA_VERSION:
        raise BoxError(f"{where}: schema_version must be {SCHEMA_VERSION}, got {v!r}")
    return int(v)


def _box_from_mapping(m, where: str) -> Box:
    if not isinstance(m, dict):
        raise BoxError(f"{where}: a box must be an object")
    missing = [k for k in FIELDS if k not in m]
    extra = sorted(set(m) - set(FIELDS))
    if missing or extra:
        raise BoxError(f"{where}: missing fields {missing}, unknown fields {extra}")
    try:
        return Box(**{k: m[k] for k in FIELDS})
    except BoxError as e:
        raise BoxError(f"{where}: {e}") from None


def dumps_json(boxes, page_width: int | None = None, page_height: int | None = None,
               page_sha256: str | None = None) -> str:
    doc = {"schema_version": SCHEMA_VERSION}
    if page_width is not None or page_height is not None or page_sha256 is not None:
        doc["page"] = {"width": page_width, "height": page_height, "sha256": page_sha256}
    doc["boxes"] = [asdict(b) for b in boxes]
    return json.dumps(doc, ensure_ascii=False, indent=2) + "\n"


def loads_json(text: str) -> BoxFile:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise BoxError(f"not valid JSON: {e}") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("boxes"), list):
        raise BoxError('a box file must be an object with "schema_version" and a "boxes" list')
    version = _check_version(doc.get("schema_version"), "box file")
    page = doc.get("page")
    if page is not None and not isinstance(page, dict):
        raise BoxError('"page" must be an object')
    boxes = tuple(_box_from_mapping(m, f"box {i}") for i, m in enumerate(doc["boxes"]))
    _unique(boxes)
    return BoxFile(version, boxes, page)


# ---------------------------------------------------------------- CSV

def dumps_csv(boxes) -> str:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_HEADER)
    for b in boxes:
        writer.writerow([SCHEMA_VERSION] + [getattr(b, k) for k in FIELDS])
    return buf.getvalue()


def loads_csv(text: str) -> BoxFile:
    rows = list(csv.reader(io.StringIO(text.lstrip("﻿"), newline="")))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows or tuple(c.strip() for c in rows[0]) != CSV_HEADER:
        raise BoxError("the CSV header must be exactly: " + ",".join(CSV_HEADER))
    boxes = []
    for n, row in enumerate(rows[1:], start=2):
        if len(row) != len(CSV_HEADER):
            raise BoxError(f"CSV row {n}: expected {len(CSV_HEADER)} columns, got {len(row)}")
        cells = dict(zip(CSV_HEADER, (c.strip() for c in row)))
        for k in ("schema_version",) + INT_FIELDS:
            if not _INT_TEXT.fullmatch(cells[k]):
                raise BoxError(f"CSV row {n}: {k} must be an integer, got {cells[k]!r}")
            cells[k] = int(cells[k])
        _check_version(cells.pop("schema_version"), f"CSV row {n}")
        boxes.append(_box_from_mapping(cells, f"CSV row {n}"))
    _unique(boxes)
    return BoxFile(SCHEMA_VERSION, tuple(boxes), None)


def _unique(boxes) -> None:
    seen = set()
    for b in boxes:
        if b.id in seen:
            raise BoxError(f"duplicate box id {b.id!r}")
        seen.add(b.id)


# ---------------------------------------------------------------- files

def _write(path, text: str, overwrite: bool) -> None:
    p = assert_path_allowed(path, write=True)
    with open(p, "w" if overwrite else "x", encoding="utf-8", newline="") as f:
        f.write(text)


def save_json(path, boxes, page_width: int | None = None, page_height: int | None = None,
              page_sha256: str | None = None, overwrite: bool = False) -> None:
    """Write a box file. An existing file is not replaced unless overwrite=True."""
    _write(path, dumps_json(boxes, page_width, page_height, page_sha256), overwrite)


def save_csv(path, boxes, overwrite: bool = False) -> None:
    _write(path, dumps_csv(boxes), overwrite)


def load_json(path) -> BoxFile:
    with open(assert_path_allowed(path), encoding="utf-8-sig") as f:
        return loads_json(f.read())


def load_csv(path) -> BoxFile:
    with open(assert_path_allowed(path), encoding="utf-8-sig", newline="") as f:
        return loads_csv(f.read())
