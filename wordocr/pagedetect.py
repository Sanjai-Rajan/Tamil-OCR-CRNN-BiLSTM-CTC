"""
Is an upload one word, one line of words, or a page? Decided from the page's own layout, before any word recognition.

The word recognizer reads ONE cropped word: a whole page squeezed to its 48-pixel input height comes out as one or two
characters, which must never be shown as the page's text. This module runs the project's automatic segmentation
(pageocr.pipeline.segment: illumination correction, Sauvola threshold, component cleaning, deskew, columns, lines,
words; classical image processing, no model) on the decoded image and counts what it finds:

    "word"      at most one substantial line holding at most one substantial word region (or no text found)
    "line"      one substantial line with several word regions: word OCR still runs (it reads the line without
                spaces), with a recommendation to use Full-page OCR
    "page"      at least 3 rows with 6 substantial word regions, or 2 rows with 8 (columns side by side share rows):
                word OCR is NOT run on the image, and there is no override; the user is sent to Full-page OCR
    "multiline" at least 2 rows with 3 regions, below the page rule: refused like a page unless the user explicitly
                confirms that the image is one word (heavily blurred or broken words can split into two "rows")

"Substantial" filters out the specks and slivers that a noisy margin or a neighbouring line's descenders can produce:
a word region counts only if it is at least MIN_WORD_PX tall and at least MIN_REL_HEIGHT of the tallest word region.
Regions are grouped into rows by vertical overlap (a blurred word that breaks into slivers stays one row), and a row
counts as a line of text only if it is comparable to the main row in height and width (pieces of a neighbouring line
that a word crop often contains are not).
The orientation or aspect ratio of the image is NOT used, so a tall crop of one word is still read. Light text on a
dark background is inverted (dark_on_light) before the segmentation looks at it.

The decision is a heuristic on segmentation tested with generated images (tests/test_fullpage_*.py); it can be wrong
on unusual images, and the reason is always shown with the counts it rests on.
"""
import time
from dataclasses import dataclass

from pageocr import pipeline
from pageocr.layout import OcrImage

MIN_WORD_PX = 8
MIN_REL_HEIGHT = 0.4
INVERT_MIN_LEVELS = 15       # light-on-dark only if the bright tail lies this many gray levels above the paper ...
INVERT_TAIL_RATIO = 1.25     # ... and reaches this much further than the dark tail
ROW_OVERLAP = 0.5           # two regions are on one row if they share this share of the smaller one's height
PAGE_RULES = ((3, 6), (2, 8))   # "page": at least (rows, word regions) of one of these pairs
MULTI_MIN_ROWS, MULTI_MIN_WORDS = 2, 3
ROW_MIN_REL_HEIGHT = 0.6    # a row counts as a line of text only if it is this tall relative to the tallest row ...
ROW_MIN_REL_EXTENT = 0.35   # ... and its word regions cover this share of the widest row's covered width
LINE_WORD_ASPECT = 1.5       # in a one-row image, only regions at least this many times wider than tall count as words ...
LINE_MIN_GAP = 0.3           # ... and only if they stand at least this many row heights apart (a word space, not a break)


@dataclass(frozen=True)
class Assessment:
    kind: str                 # "word", "line", "multiline" or "page"
    lines: int                # rows of text: substantial word regions grouped by vertical overlap
    words: int                # substantial word regions
    blocks: int               # blocks (columns / text regions) holding substantial word regions
    reason: str               # one sentence for the user
    seconds: float

    @property
    def is_page(self) -> bool:
        """Word OCR is not run on this image (a "multiline" image only if the user explicitly says it is one word)."""
        return self.kind in ("page", "multiline")

    @property
    def overridable(self) -> bool:
        return self.kind == "multiline"


def dark_on_light(gray):
    """(image with dark ink on light paper, True if it was inverted). The segmentation looks for ink darker than its
    surroundings; light text on a dark background (which the word recognizer reads, and which the training generator
    produces) would otherwise break into edge fragments. The paper is the bulk of the pixels (the median); the ink is
    the tail that departs furthest from it. If the bright tail reaches further than the dark tail, the ink is light and
    the image is inverted (255 - value: exact and reversible)."""
    import numpy as np
    lo, med, hi = np.percentile(gray, (1, 50, 99))
    # Clear evidence only: on a blank or noise-only image the two tails are equal noise, and inverting a light page
    # would hand the segmentation a dark "paper" whose noise the illumination correction amplifies into false regions.
    inverted = (hi - med) >= INVERT_MIN_LEVELS and (hi - med) > INVERT_TAIL_RATIO * (med - lo)
    return (255 - gray if inverted else gray), bool(inverted)


def rows_of(regions) -> list:
    """Word regions grouped into rows of text: a region joins a row when their vertical extents overlap by at least
    ROW_OVERLAP of the smaller height. On a page the rows are the printed lines; the slivers that a blurred, noisy
    word image breaks into overlap one another and stay one row."""
    rows = []
    for w in sorted(regions, key=lambda w: (w.box[1] + w.box[3]) / 2):
        y1, y2 = w.box[1], w.box[3]
        for r in rows:
            ov = min(r[1], y2) - max(r[0], y1)
            if ov >= ROW_OVERLAP * min(r[1] - r[0], y2 - y1):
                r[0], r[1] = min(r[0], y1), max(r[1], y2)
                r[2].append(w)
                break
        else:
            rows.append([y1, y2, [w]])
    return rows


_CACHE: dict = {}


def assess_cached(sha256: str, gray) -> Assessment:
    """assess(), remembered per file hash (the page and the word tab ask about the same upload more than once)."""
    a = _CACHE.get(sha256)
    if a is None:
        a = assess(gray)
        if len(_CACHE) >= 256:
            _CACHE.clear()
        _CACHE[sha256] = a
    return a


def refusal_message(a: Assessment) -> str:
    if a.overridable:
        return (f"{a.reason} Word OCR reads ONE cropped word per image, so it was not run. If the image really shows one "
                "word, confirm it in the Single image tab to read it anyway; otherwise use the Full-page OCR tab.")
    return (f"{a.reason} Word OCR reads ONE cropped word per image, so it was not run on this image: a page squeezed to the "
            "model's 48-pixel input height would come out as a character or two, not as the page's text. Use the "
            "Full-page OCR tab, which finds the words on the page and reads them one by one.")


def assess(gray) -> Assessment:
    """`gray`: the decoded 2-D uint8 image (wordocr.inputs.DecodedImage.gray). Reads it; changes nothing."""
    t0 = time.perf_counter()
    try:
        seg = pipeline.segment(OcrImage(gray=dark_on_light(gray)[0]))
    except Exception as e:                       # segmentation trouble must not block a word that could be read
        return Assessment("word", 0, 0, 0, f"The layout check could not run ({type(e).__name__}); the image is treated as one word.",
                          time.perf_counter() - t0)
    words = [w for w in seg.words if w.kind == "word"]
    tallest = max((w.box[3] - w.box[1] for w in words), default=0)
    floor = max(MIN_WORD_PX, MIN_REL_HEIGHT * tallest)
    big = [w for w in words if (w.box[3] - w.box[1]) >= floor and (w.box[2] - w.box[0]) >= 2]
    rows = rows_of(big)
    if rows:                    # pieces of a neighbouring line at a crop's edge are shorter and narrower than the text
        tall = max(r[1] - r[0] for r in rows)
        wide = max(sum(w.box[2] - w.box[0] for w in r[2]) for r in rows)
        rows = [r for r in rows if (r[1] - r[0]) >= ROW_MIN_REL_HEIGHT * tall
                and sum(w.box[2] - w.box[0] for w in r[2]) >= ROW_MIN_REL_EXTENT * wide]
        big = [w for r in rows for w in r[2]]
    blocks = {w.block_index for w in big}
    n_l, n_w, n_b = len(rows), len(big), len(blocks)
    secs = time.perf_counter() - t0
    found = f"{n_l} line(s), {n_w} word region(s), {n_b} text block(s) were found"
    if any(n_l >= r and n_w >= w for r, w in PAGE_RULES):
        what = "a page or several lines of text" + (" in several columns or regions" if n_b >= 2 else "")
        return Assessment("page", n_l, n_w, n_b, f"The image appears to contain {what}: {found}.", secs)
    if n_l >= MULTI_MIN_ROWS and n_w >= MULTI_MIN_WORDS:
        return Assessment("multiline", n_l, n_w, n_b, f"The image appears to contain more than one line of text ({found}); "
                          "a blurred or broken single word can look like this too.", secs)
    wordlike = sorted((w for w in big if (w.box[2] - w.box[0]) >= LINE_WORD_ASPECT * (w.box[3] - w.box[1])), key=lambda w: w.box[0])
    row_h = max((r[1] - r[0] for r in rows), default=0)
    spaced = len(wordlike) >= 2 and all(b.box[0] - a.box[2] >= LINE_MIN_GAP * row_h for a, b in zip(wordlike, wordlike[1:]))
    if spaced:
        return Assessment("line", n_l, len(wordlike), n_b, f"The image may contain one line of {len(wordlike)} words.", secs)
    if n_w == 0:
        return Assessment("word", n_l, n_w, n_b, "No separate text regions were found; the image is treated as one word.", secs)
    return Assessment("word", n_l, n_w, n_b, "One word region was found.", secs)
