"""
Export of a recognized page: plain text and JSON.

The order of the text comes only from the operator's box metadata: column_index, then line_index, then the order in
which the boxes were supplied. The recognizer contributes the words, never their order.

TXT   words of a line separated by one space, lines by a newline, columns by one blank line. A box with an empty
      prediction contributes no word; a line with no words is left out.
JSON  one record per recognized box, in the supplied order, each carrying the box id and geometry, the text, the
      confidence, the clipped flag, and the page SHA-256, checkpoint SHA-256 and engine version it was produced with.
      The confidence is uncalibrated: a flagging heuristic, not a probability of being correct.
"""
import json

from pageocr.boxes import reading_order

EXPORT_SCHEMA_VERSION = 1
STATEMENT = "V2 is an operator-assisted word-region recognition tool. It is not validated full-page OCR."
CONFIDENCE_NOTE = ("confidence = mean max-softmax probability over non-blank frames; UNCALIBRATED; "
                   "a flagging heuristic only, not a probability of being correct")


def _paired(boxes, recognitions):
    boxes, recognitions = list(boxes), list(recognitions)
    if len(boxes) != len(recognitions):
        raise ValueError(f"{len(boxes)} boxes but {len(recognitions)} recognitions")
    return boxes, recognitions


def to_txt(boxes, recognitions) -> str:
    boxes, recognitions = _paired(boxes, recognitions)
    columns: dict[int, dict[int, list[str]]] = {}
    for i in reading_order(boxes):
        if recognitions[i].text:
            columns.setdefault(boxes[i].column_index, {}).setdefault(boxes[i].line_index, []).append(recognitions[i].text)
    blocks = ["\n".join(" ".join(words) for words in lines.values()) for lines in columns.values()]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_json(boxes, recognitions, page_sha256: str, checkpoint_sha256: str, engine_version: str,
            page_name: str | None = None, page_width: int | None = None, page_height: int | None = None,
            skipped=()) -> str:
    """`skipped`: (box, reason) pairs for boxes that were not recognized (for example a crop the engine rejected)."""
    boxes, recognitions = _paired(boxes, recognitions)
    records = [{
        "box_id": b.id, "x": b.x, "y": b.y, "w": b.w, "h": b.h,
        "line_index": b.line_index, "column_index": b.column_index,
        "text": r.text, "confidence": r.confidence, "clipped": r.clipped,
        "width_px": r.width_px, "timesteps": r.timesteps,
        "page_sha256": page_sha256, "checkpoint_sha256": checkpoint_sha256, "engine_version": engine_version,
    } for b, r in zip(boxes, recognitions)]
    doc = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "statement": STATEMENT,
        "confidence_note": CONFIDENCE_NOTE,
        "engine_version": engine_version,
        "checkpoint_sha256": checkpoint_sha256,
        "page": {"name": page_name, "sha256": page_sha256, "width": page_width, "height": page_height},
        "recognitions": records,
        "skipped": [{"box_id": b.id, "x": b.x, "y": b.y, "w": b.w, "h": b.h, "line_index": b.line_index,
                     "column_index": b.column_index, "reason": str(reason)} for b, reason in skipped],
    }
    return json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
