"""
Blocks of a page in reading order: column detection by whitespace, with a horizontal-then-vertical XY cut.

The page is cut into horizontal text strips (rows of ink separated by blank rows). A column gutter is a blank vertical
band, at least gutter_min_h wide, that is shared by at least gutter_min_lines consecutive strips. Word gaps do not
line up over that many lines; gutters do. Consecutive strips with a common gutter form a multi-column group, cut at
its gutters into columns; the other strips form single-column blocks. That is one horizontal cut followed by vertical
cuts, which handles a full-width heading above two columns ("mixed layout").

Reading order of the blocks: groups top to bottom; inside a multi-column group, columns left to right.

This reads only the boolean ink mask of a layout image.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Block:
    x1: int
    y1: int
    x2: int
    y2: int                 # exclusive
    kind: str               # "single" or "column"
    group: int              # index of the horizontal group the block belongs to
    column_in_group: int    # 0 for a single block; 0, 1, ... left to right in a multi-column group


def runs(flags: np.ndarray) -> list:
    """[(start, end)] of the True runs of a 1-D boolean array, end exclusive."""
    f = np.concatenate([[False], flags.astype(bool), [False]])
    d = np.diff(f.astype(np.int8))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def strips(ink: np.ndarray, y1: int, y2: int, min_gap: int = 1) -> list:
    """Text strips between rows y1 and y2: runs of rows that hold ink, joined across blank gaps narrower than min_gap."""
    rows = ink[y1:y2].any(axis=1)
    out = []
    for a, b in runs(rows):
        if out and a - out[-1][1] < min_gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return [(a + y1, b + y1) for a, b in out]


def _blank_intervals(ink: np.ndarray, band: tuple, x1: int, x2: int) -> list:
    cols = ~ink[band[0]:band[1], x1:x2].any(axis=0)
    return [(a + x1, b + x1) for a, b in runs(cols)]


def _intersect(a: list, b: list, min_width: float) -> list:
    out = []
    for p in a:
        for q in b:
            lo, hi = max(p[0], q[0]), min(p[1], q[1])
            if hi - lo >= min_width:
                out.append((lo, hi))
    return out


def find_blocks(ink: np.ndarray, h: float, gutter_min_h: float = 1.0, gutter_min_lines: int = 4) -> tuple:
    """(blocks in reading order, flags). `h` is the page's text height (LayoutImage.h)."""
    if ink.dtype != np.bool_ or ink.ndim != 2:
        raise TypeError("find_blocks takes the boolean ink mask of a layout image")
    flags = []
    ys, xs = np.nonzero(ink.any(axis=1))[0], np.nonzero(ink.any(axis=0))[0]
    if len(ys) == 0:
        return [], ["no text found"]
    X1, X2, Y1, Y2 = int(xs[0]), int(xs[-1]) + 1, int(ys[0]), int(ys[-1]) + 1
    st = strips(ink, Y1, Y2)
    raw = [_blank_intervals(ink, s, X1, X2) for s in st]
    # A gutter is wider than the page's word spaces. On a widely spaced or justified page the word spaces themselves can be
    # wide and can line up over a few lines; so the minimum gutter width follows the page's own larger word gaps.
    # (A thin strip is a row of marks above a line, not a line: it says nothing about word spaces or about columns.)
    is_text = [s[1] - s[0] >= 0.6 * h for s in st]
    wide = [iv[1] - iv[0] for b, t in zip(raw, is_text) if t for iv in b if iv[1] - iv[0] >= 0.3 * h and iv[0] > X1 and iv[1] < X2]
    gmin = max(3.0, gutter_min_h * h)
    clear = max(gmin, 1.6 * float(np.percentile(wide, 75)) if wide else 0.0)      # clearly wider than the page's word spaces
    # interior blank bands of every strip: a margin beside a short line is not a gutter, so the page's own margins are cut off
    blanks = [[iv for iv in b if iv[1] - iv[0] >= gmin] for b in raw]
    groups, i = [], 0
    while i < len(st):
        common, j = blanks[i], i + 1
        while j < len(st) and common:
            nxt = _intersect(common, blanks[j], gmin)
            if not nxt:
                break
            common, j = nxt, j + 1
        # a gutter must be interior: text on both sides of it, somewhere in the group
        real = []
        n_text = sum(is_text[i:j])
        if n_text >= gutter_min_lines:
            for lo, hi in common:
                # a band no wider than the page's word spaces must persist over twice as many lines to count as a gutter
                if hi - lo < clear and n_text < 2 * gutter_min_lines:
                    continue
                left = ink[st[i][0]:st[j - 1][1], X1:lo].any(axis=1)
                right = ink[st[i][0]:st[j - 1][1], hi:X2].any(axis=1)
                n = st[j - 1][1] - st[i][0]
                if left.sum() >= 0.3 * n and right.sum() >= 0.3 * n:
                    real.append((lo, hi))
        if real:
            groups.append(("columns", i, j, real))
            i = j
        else:
            if groups and groups[-1][0] == "single":
                groups[-1] = ("single", groups[-1][1], i + 1, [])
            else:
                groups.append(("single", i, i + 1, []))
            i += 1
    blocks = []
    for g, (kind, a, b, gutters) in enumerate(groups):
        y1, y2 = st[a][0], st[b - 1][1]
        if kind == "single":
            cols = np.nonzero(ink[y1:y2].any(axis=0))[0]
            blocks.append(Block(int(cols[0]), y1, int(cols[-1]) + 1, y2, "single", g, 0))
            continue
        edges = [X1] + [v for lo, hi in sorted(gutters) for v in (lo, hi)] + [X2]
        c = 0
        for k in range(0, len(edges), 2):
            sub = ink[y1:y2, edges[k]:edges[k + 1]]
            if not sub.any():
                continue
            cols, rows = np.nonzero(sub.any(axis=0))[0], np.nonzero(sub.any(axis=1))[0]
            blocks.append(Block(edges[k] + int(cols[0]), y1 + int(rows[0]), edges[k] + int(cols[-1]) + 1, y1 + int(rows[-1]) + 1, "column", g, c))
            c += 1
    if sum(1 for k in groups if k[0] == "columns") > 1:
        flags.append("more than one multi-column region: mixed layout")
    if any(k[0] == "columns" for k in groups) and any(k[0] == "single" for k in groups):
        flags.append("full-width text and columns on one page: mixed layout")
    return blocks, flags
