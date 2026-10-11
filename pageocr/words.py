"""
Word boxes of a text line. Word level only: nothing here separates characters.

  1. The line's connected components are sorted left to right.
  2. Components that overlap horizontally are one group: a pulli over its consonant, a vowel sign under a letter,
     the two parts of a broken glyph.
  3. A small mark (pulli, dot, superscript) that stands just beside a group, and above the middle of the line, is
     joined to the nearest group, so it cannot become a "word" of its own.
  4. The gaps between groups decide the words. Each line sets its own threshold when it can: if its gaps fall into two
     clearly separated sizes, the threshold is half-way between them. A line with too few gaps, or gaps of one size
     only, uses the page's threshold (pooled from the lines that could decide), or a default tied to the line height.
     A line set solid, with no word spaces at all, therefore stays one run; that is intended.
  5. Trailing punctuation low on the line joins the word before it when it stands close; otherwise it is a unit of
     its own. Merging is conservative: when in doubt the gap is kept.

Every box is the tight box of its ink, in the pixel frame of the layout image (which is the frame of the OCR image).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Word:
    x1: int
    y1: int
    x2: int
    y2: int                  # exclusive, tight to the ink
    components: tuple        # indices into the layout image's component array
    kind: str                # "word" or "punctuation"
    gap_before: int          # pixels of white between this word and the one before it on the line; -1 for the first


def _groups(components: np.ndarray, idx: list) -> list:
    """Components merged by horizontal overlap. Each group: [x1, y1, x2, y2, [component indices]]."""
    out = []
    for i in sorted(idx, key=lambda i: (components[i][0], components[i][1])):
        c = components[i]
        if out and c[0] < out[-1][2]:                      # overlaps the running group in x
            g = out[-1]
            g[1], g[2], g[3] = min(g[1], c[1]), max(g[2], c[2]), max(g[3], c[3])
            g[4].append(i)
        else:
            out.append([int(c[0]), int(c[1]), int(c[2]), int(c[3]), [i]])
    return out


def _attach_marks(groups: list, line_h: float, y_mid: float, mark_side: float, mark_attach: float) -> list:
    """Join small high marks to the nearest neighbouring group when they stand within mark_attach x line height."""
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for i, g in enumerate(groups):
            w, h = g[2] - g[0], g[3] - g[1]
            if max(w, h) > mark_side * line_h or (g[1] + g[3]) / 2.0 >= y_mid:
                continue                                   # not a small mark in the upper half
            left = g[0] - groups[i - 1][2] if i > 0 else None
            right = groups[i + 1][0] - g[2] if i + 1 < len(groups) else None
            cand = [(d, j) for d, j in ((left, i - 1), (right, i + 1)) if d is not None and d <= mark_attach * line_h]
            if not cand:
                continue
            _, j = min(cand)
            t = groups[j]
            t[0], t[1], t[2], t[3] = min(t[0], g[0]), min(t[1], g[1]), max(t[2], g[2]), max(t[3], g[3])
            t[4].extend(g[4])
            del groups[i]
            changed = True
            break
    return groups


def line_threshold(gaps: list, line_h: float, separation: float, floor_h: float):
    """The line's own word-gap threshold, or None if its gaps do not fall into two clearly separated sizes."""
    g = sorted(v for v in gaps if v > 0)
    if len(g) < 3:
        return None
    best, cut = 0.0, None
    for k in range(1, len(g)):                             # two-class split maximising the between-class variance
        a, b = g[:k], g[k:]
        score = len(a) * len(b) * (sum(b) / len(b) - sum(a) / len(a)) ** 2
        if score > best:
            best, cut = score, k
    if cut is None:
        return None
    lo_max, hi_min = g[cut - 1], g[cut]
    if hi_min < separation * max(lo_max, 1) or hi_min < floor_h * line_h:
        return None
    return (lo_max + hi_min) / 2.0


def find_words(components: np.ndarray, line, params, page_threshold_h=None) -> tuple:
    """(words of the line, diagnostics). `params` is a pageocr.layout.SegParams."""
    line_h = float(line.y2 - line.y1)
    y_mid = (line.y1 + line.y2) / 2.0
    groups = _attach_marks(_groups(components, list(line.components)), line_h, y_mid, params.mark_side_h, params.mark_attach_h)
    if not groups:
        return [], {"threshold_px": None, "threshold_source": "none", "gaps": []}
    gaps = [groups[i + 1][0] - groups[i][2] for i in range(len(groups) - 1)]
    own = line_threshold(gaps, line_h, params.gap_separation, params.gap_floor_h)
    if own is not None:
        thr, source = own, "line"
    elif page_threshold_h is not None:
        thr, source = page_threshold_h * line_h, "page"
    else:
        thr, source = params.gap_default_h * line_h, "default"
    thr = max(thr, params.gap_floor_h * line_h)
    # punctuation: a small group low on the line. It ends the word before it if close; it never starts a word.
    def is_punct(g):
        return max(g[2] - g[0], g[3] - g[1]) <= params.mark_side_h * line_h and (g[1] + g[3]) / 2.0 >= y_mid
    words, cur, cur_gap = [], None, -1
    for i, g in enumerate(groups):
        gap = gaps[i - 1] if i > 0 else -1
        join = cur is not None and (gap < thr or (is_punct(g) and gap < params.punctuation_attach * thr))
        if join:
            cur[0], cur[1], cur[2], cur[3] = min(cur[0], g[0]), min(cur[1], g[1]), max(cur[2], g[2]), max(cur[3], g[3])
            cur[4].extend(g[4])
        else:
            if cur is not None:
                words.append((cur, cur_gap))
            cur, cur_gap = [g[0], g[1], g[2], g[3], list(g[4])], gap
    words.append((cur, cur_gap))
    out = []
    for w, gap in words:
        only_punct = all(max(components[i][2] - components[i][0], components[i][3] - components[i][1]) <= params.mark_side_h * line_h for i in w[4])
        out.append(Word(int(w[0]), int(w[1]), int(w[2]), int(w[3]), tuple(sorted(w[4])), "punctuation" if only_punct else "word", int(gap)))
    return out, {"threshold_px": float(thr), "threshold_source": source, "gaps": [int(v) for v in gaps], "own_threshold_h": None if own is None else own / line_h}


def page_threshold(components: np.ndarray, lines: list, params):
    """The pooled word-gap threshold of a page, in units of line height: the median of the lines that decided for themselves."""
    vals = []
    for ln in lines:
        line_h = float(ln.y2 - ln.y1)
        groups = _attach_marks(_groups(components, list(ln.components)), line_h, (ln.y1 + ln.y2) / 2.0, params.mark_side_h, params.mark_attach_h)
        gaps = [groups[i + 1][0] - groups[i][2] for i in range(len(groups) - 1)]
        own = line_threshold(gaps, line_h, params.gap_separation, params.gap_floor_h)
        if own is not None and line_h > 0:
            vals.append(own / line_h)
    return float(np.median(vals)) if len(vals) >= 3 else None
