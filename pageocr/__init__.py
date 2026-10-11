"""
pageocr - V2: operator-assisted word-region recognition ("manual region OCR").

An operator supplies word boxes on a page image; each box is cropped from the ORIGINAL grayscale page and read by the
frozen Experiment 002-A word recognizer. This package does not find text, lines, columns or reading order by itself,
and it is not validated full-page OCR.

Modules
    engine   OcrEngine: the frozen recognizer behind a strict input contract (no Streamlit)
    boxes    Box schema, JSON/CSV files, validation, cropping, preview <-> original coordinate mapping
    imageio  load_page: one decode to 2-D uint8 grayscale, with its SHA-256; the preview derived from it
    export   to_txt / to_json
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:               # the recognizer's code lives in the project's src/ package
    sys.path.insert(0, str(ROOT))

ENGINE_VERSION = "pageocr-2.0.0"

# Folder names that hold frozen research data. No page, box file or checkpoint is read from, and nothing is written
# to, a path that has one of these names anywhere in it (so a copy of such a folder is refused as well).
PROTECTED_NAMES = frozenset({"real_docs", "r1_t4b_review", "r1_pilot",
                             "r1_t4c_review", "r1_t4d_review", "r1_s012_amendment"})    # R1 records outside the repo (2026-10-10)
# Project folders this package never writes into.
READ_ONLY_PROJECT_DIRS = ("src", "tools", "configs", "runs", "data", "real_docs", "legacy", "reports", "pageocr",
                          "development_pages/original", "development_pages/annotations", "development_pages/manifests",
                          "development_pages/previews", "development_pages/experiments")


class ProtectedPathError(ValueError):
    """The path points into frozen research data (or, for writing, into a read-only project folder)."""


def assert_path_allowed(path, write: bool = False) -> Path:
    """Return the resolved path, or raise ProtectedPathError. Pure: the file system is not touched beyond resolving."""
    p = Path(path).resolve()
    hit = PROTECTED_NAMES.intersection(part.casefold() for part in p.parts)
    if hit:
        raise ProtectedPathError(f"Refused: {p} is inside protected research data ({sorted(hit)[0]}).")
    if write:
        # A device-namespace (\\?\, \\.\) or network (\\server\share, \\localhost\C$) spelling can name a project folder
        # without matching it textually, so such paths are never written to.
        if str(p).startswith(("\\\\", "//")):
            raise ProtectedPathError(f"Refused: {p} is a device-namespace or network path; this package writes to local paths only.")
        for name in READ_ONLY_PROJECT_DIRS:
            base = ROOT / name
            if p == base or base in p.parents:
                raise ProtectedPathError(f"Refused: {p} is inside the read-only project folder {name}/.")
    return p
