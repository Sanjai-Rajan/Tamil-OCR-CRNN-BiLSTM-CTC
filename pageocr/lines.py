"""
Text lines of a block, top to bottom.

A line is a band of rows that hold ink. Three corrections are applied to the raw bands:
  - a thin band (a row of pulli, dots or other marks that a blank row separates from its letters) is joined to the
    band it belongs to: the band below if that is closer, as for marks above letters, else the band above;
  - a band much taller than the others is two or more lines that touch; it is cut at the weakest rows of its ink
    profile, and the cut is flagged, because a cut through touching print is a guess;
  - components are then assigned to the line that holds most of their height.

Reads the boolean ink mask and the component boxes of a layout image only.
"""
from dataclasses import dataclass

import numpy as np

from pageocr.columns import runs


@dataclass(frozen=True)
class Line:
    y1: int
    y2: int                  # exclusive; the band of rows of this line
    components: tuple        # indices into the layout image's component array
    forced_split: bool       # True if this line came from cutting a band of touching lines


def _bands(ink: np.ndarray, block) -> list:
    rows = ink[block.y1:block.y2, block.x1:block.x2].any(axis=1)
    return [[a + block.y1, b + block.y1, False] for a, b in runs(rows)]


def find_lines(ink: np.ndarray, components: np.ndarray, block, split_factor: float = 1.75, thin_factor: float = 0.45) -> list:
    bands = _bands(ink, block)
    if not bands:
        return []
    heights = np.array([b[1] - b[0] for b in bands], float)
    med = float(np.median(heights[heights >= np.percentile(heights, 50)])) if len(heights) else 0.0
    # join thin bands of marks to a neighbour
    changed = True
    while changed and len(bands) > 1:
        changed = False
        for i, b in enumerate(bands):
            if b[1] - b[0] >= thin_factor * med:
                continue
            up = b[0] - bands[i - 1][1] if i > 0 else None
            down = bands[i + 1][0] - b[1] if i + 1 < len(bands) else None
            if down is not None and (up is None or down <= up):
                bands[i + 1][0] = b[0]
            elif up is not None:
                bands[i - 1][1] = b[1]
            else:
                continue
            del bands[i]
            changed = True
            break
    # cut bands of touching lines at the weakest rows
    out = []
    for y1, y2, _ in bands:
        n = int(round((y2 - y1) / med)) if med > 0 else 1
        if med > 0 and (y2 - y1) >= split_factor * med and n >= 2:
            prof = ink[y1:y2, block.x1:block.x2].sum(axis=1).astype(float)
            k = max(3, int(med * 0.15) | 1)
            smooth = np.convolve(prof, np.ones(k) / k, mode="same")
            cuts, step = [], (y2 - y1) / n
            for j in range(1, n):
                c = int(round(j * step))
                lo, hi = max(1, int(c - 0.3 * step)), min(len(smooth) - 1, int(c + 0.3 * step))
                cuts.append(y1 + lo + int(np.argmin(smooth[lo:hi]))) if hi > lo else cuts.append(y1 + c)
            edges = [y1] + cuts + [y2]
            out.extend([edges[j], edges[j + 1], True] for j in range(len(edges) - 1) if edges[j + 1] > edges[j])
        else:
            out.append([y1, y2, False])
    # assign components to the line holding most of their height
    members = [[] for _ in out]
    for idx, c in enumerate(components):
        if c[2] <= block.x1 or c[0] >= block.x2 or c[3] <= block.y1 or c[1] >= block.y2:
            continue
        cx = (c[0] + c[2]) / 2.0
        if not (block.x1 <= cx < block.x2):
            continue
        best, share = None, 0
        for li, (y1, y2, _) in enumerate(out):
            ov = min(c[3], y2) - max(c[1], y1)
            if ov > share:
                best, share = li, ov
        if best is not None:
            members[best].append(idx)
    return [Line(int(y1), int(y2), tuple(m), bool(f)) for (y1, y2, f), m in zip(out, members) if m]
