"""
Reading order of an automatically segmented page.

Blocks come in order from pageocr.columns (groups top to bottom; columns of a group left to right). Inside a block,
lines run top to bottom and words left to right. Printed line breaks are kept: a word never moves to another line and
lines are never joined. The result uses the project's existing box convention, so pageocr.export can write it:
column_index is the block's position in reading order, line_index the line's position inside its block.

The order comes from geometry only. The recognizer plays no part in it.
"""
from dataclasses import dataclass

from pageocr.boxes import Box


@dataclass(frozen=True)
class PlacedWord:
    id: str                  # "b00_l003_w02": block, line in block, word in line
    block_index: int
    line_index: int
    word_index: int
    box: tuple               # (x1, y1, x2, y2), exclusive, in the frame of the OCR image
    box_original: tuple      # the same word in ORIGINAL page pixels (differs only if the page was deskewed)
    kind: str                # "word" or "punctuation"
    gap_before: int
    line_forced_split: bool  # the line was cut out of a band of touching lines
    block_kind: str          # "single" or "column"


def place(blocks: list, lines_by_block: list, words_by_line: list, ocr) -> list:
    """blocks[i] -> lines_by_block[i][j] -> words_by_line[i][j]: flattened into reading order."""
    out = []
    for bi, block in enumerate(blocks):
        for li, line in enumerate(sorted(range(len(lines_by_block[bi])), key=lambda k: lines_by_block[bi][k].y1)):
            ln = lines_by_block[bi][line]
            for wi, w in enumerate(sorted(words_by_line[bi][line], key=lambda w: (w.x1, w.y1))):
                box = (w.x1, w.y1, w.x2, w.y2)
                out.append(PlacedWord(f"b{bi:02d}_l{li:03d}_w{wi:02d}", bi, li, wi, box, tuple(ocr.box_in_original(box)), w.kind, w.gap_before,
                                      ln.forced_split, block.kind))
    return out


def to_boxes(placed: list, frame: str = "ocr") -> list:
    """pageocr.boxes.Box objects, in reading order. frame "ocr": for cropping the OCR image. frame "original": for
    export and for comparison with boxes drawn on the original page."""
    if frame not in ("ocr", "original"):
        raise ValueError("frame is 'ocr' or 'original'")
    out = []
    for p in placed:
        x1, y1, x2, y2 = p.box if frame == "ocr" else p.box_original
        out.append(Box(id=p.id, x=int(x1), y=int(y1), w=int(x2 - x1), h=int(y2 - y1), line_index=p.line_index, column_index=p.block_index))
    return out


def as_text_lines(placed: list, texts: list) -> list:
    """[[block index, [line of words, ...]], ...]: the page as blocks of printed lines, from one text per placed word."""
    if len(placed) != len(texts):
        raise ValueError(f"{len(placed)} words but {len(texts)} texts")
    blocks = {}
    for p, t in zip(placed, texts):
        if t:
            blocks.setdefault(p.block_index, {}).setdefault(p.line_index, []).append(t)
    return [[b, [" ".join(words) for _, words in sorted(lines.items())]] for b, lines in sorted(blocks.items())]
