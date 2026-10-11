"""
Exports of displayed results: TXT, JSON and CSV, as bytes (UTF-8). Built from the same ItemResult objects the page shows.

TXT   one image: the recognized text and a newline. Several images: one line per image, "<file name><TAB><text>";
      an image that was not recognized reads "<file name><TAB>[not recognized: <reason>]".
JSON  every field, the model and checkpoint identity, the scope statement and the note on the score. Text is exact.
CSV   one row per image, UTF-8 with a byte-order mark so spreadsheet programs show Tamil correctly. A cell that starts
      with = + - @ or a tab/carriage return is prefixed with an apostrophe so a spreadsheet does not run it as a
      formula; the TXT and JSON exports keep the text exactly.
Nothing is written to disk here; the browser saves the bytes.
"""
import csv
import io
import json
from datetime import datetime

from wordocr import SCOPE_STATEMENT, VERSION
from wordocr.inputs import safe_stem

SCHEMA = "wordocr-results-1"
SCORE_NOTE = ("uncalibrated_score = mean, over the output frames whose top class is not the CTC blank, of that class's "
              "softmax probability. It has not been compared with correctness on any data: it is NOT a probability of "
              "being correct and NOT an accuracy. Use it only to decide which results to check first.")
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _status(r) -> str:
    return "error" if not r.ok else ("empty" if r.text == "" else "ok")


def record(r) -> dict:
    img = r.image
    return {"index": r.index, "file_name": r.name, "status": _status(r), "text": r.text if r.ok else None,
            "error": r.error or None,
            "file_sha256": img.sha256 if img else None, "format": img.format if img else None,
            "width_px": img.width if img else None, "height_px": img.height if img else None,
            "model_input_width_px": img.model_input_width if img else None, "clipped": img.clipped if img else None,
            "warnings": list(img.warnings) if img else [],
            "uncalibrated_score": r.uncalibrated_score, "timesteps": r.timesteps}


def to_json(results, model_info: dict, created: str | None = None) -> bytes:
    doc = {"schema": SCHEMA, "tool": VERSION, "created": created or datetime.now().astimezone().isoformat(timespec="seconds"),
           "scope": SCOPE_STATEMENT, "output": "raw model prediction (greedy CTC); no dictionary, language model or correction",
           "score_note": SCORE_NOTE, "model": model_info, "results": [record(r) for r in results]}
    return (json.dumps(doc, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def to_txt(results) -> bytes:
    results = list(results)
    if len(results) == 1 and results[0].ok:
        return (results[0].text + "\n").encode("utf-8")
    lines = []
    for r in results:
        name = r.name.replace("\t", " ").replace("\n", " ")
        lines.append(f"{name}\t{r.text}" if r.ok else f"{name}\t[not recognized: {r.error}]")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _cell(v) -> str:
    s = "" if v is None else str(v)
    return "'" + s if s.startswith(_FORMULA_START) else s


def to_csv(results) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(["index", "file_name", "status", "text", "error", "width_px", "height_px", "clipped", "uncalibrated_score",
                "file_sha256"])
    for r in results:
        rec = record(r)
        w.writerow([_cell(rec[k]) for k in ("index", "file_name", "status", "text", "error", "width_px", "height_px",
                                            "clipped", "uncalibrated_score", "file_sha256")])
    return ("﻿" + buf.getvalue()).encode("utf-8")


def download_name(results, ext: str) -> str:
    """'<stem>_ocr.<ext>' for one image, 'tamil_ocr_batch_<n>.<ext>' for several. ASCII-safe on every platform."""
    results = list(results)
    if len(results) == 1:
        return f"{safe_stem(results[0].name)}_ocr.{ext}"
    return f"tamil_ocr_batch_{len(results)}.{ext}"
