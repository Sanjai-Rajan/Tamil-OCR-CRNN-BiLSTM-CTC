"""
Exports of a full-page result (wordocr.fullpage.PageResult), as UTF-8 bytes built from the same object the page shows.

TXT   the reconstructed page text exactly as displayed, and a final newline.
JSON  page identity (file name, SHA-256, size), model and checkpoint identity, the settings and the crop margin, the
      segmentation summary (pipeline version, skew, text height, flags, counts, steps, parameters), the page text, and
      one record per word region in reading order (id, block, line, word, kind, box in original page pixels, box in
      the deskewed frame, status, raw text, error, uncalibrated score). The text is exact.
CSV   one row per word region, UTF-8 with a byte-order mark for spreadsheets; cells that start with = + - @ (or a
      tab / carriage return) get a leading apostrophe so a spreadsheet does not run them (wordocr.export._cell);
      the TXT and JSON exports keep the text exactly.
Nothing is written to disk here.
"""
import csv
import io
import json
from datetime import datetime

from wordocr import SCOPE_STATEMENT, VERSION
from wordocr.export import SCORE_NOTE, _cell
from wordocr.inputs import safe_stem

SCHEMA = "wordocr-page-results-1"
PAGE_STATEMENT = ("Full-page OCR = automatic page segmentation (classical image processing) + the word recognizer applied "
                  "to each word region + text rebuilt from the layout. The segmentation has been tested on generated pages "
                  "only; no accuracy has been measured on real printed pages.")
CSV_FIELDS = ["file_name", "sequence", "region_id", "block", "line", "word", "kind", "x1", "y1", "x2", "y2", "status",
              "text", "error", "uncalibrated_score"]


def word_record(w) -> dict:
    return {"sequence": w.index + 1, "region_id": w.id, "block": w.block, "line": w.line, "word": w.word, "kind": w.kind,
            "box": list(w.box), "box_deskewed_frame": list(w.box_ocr), "status": w.status, "text": w.text if w.ok else None,
            "error": w.error or None, "uncalibrated_score": w.uncalibrated_score, "crop_size": [w.crop_width, w.crop_height]}


def to_txt(result) -> bytes:
    return (result.text + "\n").encode("utf-8")


def to_json(result, created: str | None = None) -> bytes:
    doc = {"schema": SCHEMA, "tool": VERSION, "created": created or datetime.now().astimezone().isoformat(timespec="seconds"),
           "statement": PAGE_STATEMENT, "word_model_scope": SCOPE_STATEMENT,
           "output": "raw model predictions (greedy CTC) joined by the layout; no dictionary, language model or correction",
           "score_note": SCORE_NOTE,
           "page": {"file_name": result.name, "sha256": result.sha256, "width_px": result.width, "height_px": result.height},
           "model": result.model, "settings": result.settings, "crop_margin_px": result.margin_px,
           "segmentation": result.segmentation, "counts": result.counts, "seconds": result.seconds,
           "text": result.text, "words": [word_record(w) for w in result.words]}
    return (json.dumps(doc, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def to_csv(result) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(CSV_FIELDS)
    for x in result.words:
        x1, y1, x2, y2 = x.box
        row = [result.name, x.index + 1, x.id, x.block, x.line, x.word, x.kind, x1, y1, x2, y2, x.status,
               x.text if x.ok else "", x.error, x.uncalibrated_score]
        w.writerow([_cell(v) for v in row])
    return ("﻿" + buf.getvalue()).encode("utf-8")


def download_name(result, ext: str) -> str:
    return f"{safe_stem(result.name, 'page')}_page_ocr.{ext}"
