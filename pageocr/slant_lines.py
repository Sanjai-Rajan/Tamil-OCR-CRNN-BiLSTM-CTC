"""
EXPERIMENTAL (opt-in, SegParams.exp_angle_lines, off by default): text lines found by FOLLOWING the letters along their
own direction, for pages whose lines are not all parallel (photographed notebooks, perspective, hand-ruled lines).

The default pageocr.lines finds lines as horizontal bands of ink rows. On a page whose lines have different angles,
neighbouring lines overlap in rows and the bands merge (measured on generated pages: 6 lines at 0, +-1.5 and 3 degrees
became 3 bands and 3 "words"). This module replaces only the LINE step:

  1. Components of the block are sorted into bodies, marks and rule-like ones (pageocr.line_geometry._classify).
  2. Bodies are swept left to right. Each body joins the open line whose predicted centre height at the body's x is
     nearest, if within TRACK_TOL_H x text height and the line ended no more than MAX_GAP_H x text height before; the
     prediction is a Theil-Sen line through the line's last bodies (flat until it has two). Otherwise it opens a line.
  3. Fragments of one line (split by a long gap) are joined when one continues the other's prediction.
  4. Marks (pulli, dots, detached vowel-sign parts) and rule-like components join the line whose predicted centre at
     their x is nearest; a mark above a line is measured to the TOP of that line's letters, so a pulli is not handed to
     the line above.
  5. Words: the unchanged pageocr.words.find_words runs on a LEVELLED copy of the line's component boxes (each box
     shifted vertically by -slope x (x - x_ref): a shear of box coordinates, no pixel is touched). The word boxes are
     then rebuilt as the union of the members' ORIGINAL boxes. Crops are therefore ordinary axis-aligned crops of the
     original grayscale OCR image: nothing is rotated or resampled for the recognizer.

The Line objects returned carry the axis-aligned band of their members (y1, y2) in the OCR frame, sorted by y1, so the
rest of the pipeline (reading order, ids, diagnostics) works unchanged.
"""
import numpy as np

from pageocr.line_geometry import _classify, _theil_sen
from pageocr.lines import Line
from pageocr.words import Word, find_words, page_threshold

TRACK_TOL_H = 0.5
MAX_GAP_H = 8.0
FIT_LAST = 12
MERGE_RESID_H = 0.3


class _Track:
    def __init__(self, i, x, y, x2):
        self.idx, self.xs, self.ys, self.x2 = [i], [x], [y], x2

    def fit(self, h):
        xs, ys = np.array(self.xs[-FIT_LAST:]), np.array(self.ys[-FIT_LAST:])
        if len(xs) >= 3:
            f = _theil_sen(xs, ys, 1.0 * h)
            if f is not None:
                return f[0], f[1]
        if len(xs) >= 2 and xs[-1] - xs[0] >= 1.5 * h:
            s = float(np.clip((ys[-1] - ys[0]) / (xs[-1] - xs[0]), -0.12, 0.12))
            return s, float(np.median(ys - s * xs))
        return 0.0, float(np.median(ys))

    def pred(self, x, h):
        s, b = self.fit(h)
        return s * x + b


def _track(comps, bodies, h):
    order = sorted(bodies, key=lambda i: ((comps[i][0] + comps[i][2]) / 2.0, comps[i][1]))
    tracks = []
    for i in order:
        c = comps[i]
        x, y = (c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0
        best, bd = None, None
        for t in tracks:
            if x - t.x2 > MAX_GAP_H * h:
                continue
            d = abs(y - t.pred(x, h))
            if d <= TRACK_TOL_H * h and (bd is None or d < bd):
                best, bd = t, d
        if best is None:
            tracks.append(_Track(i, x, y, c[2]))
        else:
            best.idx.append(i); best.xs.append(x); best.ys.append(y); best.x2 = max(best.x2, c[2])
    return _consolidate(tracks, h)


def _fit_all(xs, ys, h):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) >= 3:
        f = _theil_sen(xs, ys, 1.0 * h)
        if f is not None:
            return f[0], f[1]
    return 0.0, float(np.median(ys))


def _consolidate(tracks, h):
    """Merge tracks that are one line: fragments split by a gap, and parallel tracks that shared one line's letters.
    Two tracks merge when one straight (Theil-Sen) fit through all their points leaves a median residual <= MERGE_RESID_H
    x text height for EACH of them, and they are no more than MAX_GAP_H x text height apart along x. Neighbouring real
    lines are >= ~1.5 x text height apart, so their union never fits."""
    changed = True
    while changed and len(tracks) > 1:
        changed = False
        best = None
        for i, a in enumerate(tracks):
            ya = float(np.median(a.ys))
            for b in tracks[i + 1:]:
                gap = max(min(b.xs) - max(a.xs), min(a.xs) - max(b.xs), 0.0)        # 0 when they overlap along x
                if gap > MAX_GAP_H * h:
                    continue
                sa, ba = _fit_all(a.xs, a.ys, h)
                if abs(sa * float(np.median(b.xs)) + ba - float(np.median(b.ys))) > 1.2 * h and abs(ya - float(np.median(b.ys))) > 1.2 * h:
                    continue                                    # far apart vertically: cannot be one line
                xs, ys = a.xs + b.xs, a.ys + b.ys
                s, c = _fit_all(xs, ys, h)
                ra = float(np.median(np.abs(np.array(a.ys) - (s * np.array(a.xs) + c))))
                rb = float(np.median(np.abs(np.array(b.ys) - (s * np.array(b.xs) + c))))
                score = max(ra, rb)
                if score <= MERGE_RESID_H * h and (best is None or score < best[0]):
                    best = (score, a, b)
        if best is not None:
            _, a, b = best
            a.idx += b.idx; a.xs += b.xs; a.ys += b.ys; a.x2 = max(a.x2, b.x2)
            order = np.argsort(a.xs, kind="stable")
            a.idx, a.xs, a.ys = [a.idx[k] for k in order], [a.xs[k] for k in order], [a.ys[k] for k in order]
            tracks.remove(b)
            changed = True
    return tracks


def find_lines_tracked(components: np.ndarray, block, h: float) -> list:
    """[(Line, slope, x_ref)] of a block, sorted by the top of each line's band. Line.components index `components`."""
    inside = [i for i, c in enumerate(components)
              if c[2] > block.x1 and c[0] < block.x2 and c[3] > block.y1 and c[1] < block.y2 and block.x1 <= (c[0] + c[2]) / 2.0 < block.x2]
    if not inside:
        return []
    sub = components[np.array(inside)]
    b, m, r = _classify(sub, h)
    bodies = [inside[k] for k in b]
    others = [inside[k] for k in np.r_[m, r].astype(int)]
    tracks = _track(components, bodies, h) if bodies else []
    if not tracks:                                  # no letter bodies: one line of whatever is there
        idx = tuple(sorted(inside))
        c = components[np.array(idx)]
        return [(Line(int(c[:, 1].min()), int(c[:, 3].max()), idx, False), 0.0, float(block.x1))]
    fits = [_fit_all(t.xs, t.ys, h) for t in tracks]              # a line's final direction uses all its points
    tops = []
    for t, (s, b0) in zip(tracks, fits):                 # top of the letters relative to the centre line
        cc = components[np.array(t.idx)]
        tops.append(float(np.median(cc[:, 1] - (s * (cc[:, 0] + cc[:, 2]) / 2.0 + b0))))
    members = [list(t.idx) for t in tracks]
    for i in others:
        c = components[i]
        x, y = (c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0
        best, bd = None, None
        for k, (t, (s, b0)) in enumerate(zip(tracks, fits)):
            if not (t.xs[0] - 3 * h <= x <= t.x2 + 3 * h):
                continue
            centre = s * x + b0
            top = centre + tops[k]
            d = (top - c[3]) if c[3] <= top else abs(y - centre)        # a mark above: its distance to the letters' top
            d = max(d, 0.0)
            if bd is None or d < bd:
                best, bd = k, d
        if best is None:                                  # nowhere near any line: nearest centre over the whole block
            best = int(np.argmin([abs(y - (s * x + b0)) for s, b0 in fits]))
        members[best].append(i)
    out = []
    for mem, (s, b0) in zip(members, fits):
        c = components[np.array(mem)]
        out.append((Line(int(c[:, 1].min()), int(c[:, 3].max()), tuple(sorted(mem)), False), float(s), float(block.x1)))
    out.sort(key=lambda t: (t[0].y1, t[0].y2))
    return out


MIN_TRACK_BODIES = 4
HEALTHY_SHARE = 0.8


def prefer_tracked(components: np.ndarray, band_lines: list, tracked: list, h: float) -> tuple:
    """(use the tracked lines?, reason) for one block. The tracked lines replace the horizontal bands only when
      - the bands are evidently wrong: one band holds the letter bodies of two or more substantial tracked lines
        (>= MIN_TRACK_BODIES bodies each), i.e. lines of different direction were merged into one band; and
      - the tracking is healthy: the block's small fragments do not outnumber its letter bodies (broken, faint print
        is mostly fragments, and the text-height unit then underestimates the letters), and >= HEALTHY_SHARE of the
        letter bodies lie on substantial tracked lines; there the bands are the safer choice.
    Otherwise the block keeps the default horizontal bands, unchanged."""
    if not tracked or not band_lines:
        return False, "no lines"
    every = np.array(sorted({int(j) for ln, _, _ in tracked for j in ln.components}))
    nb, nm, _ = (len(v) for v in _classify(components[every], h))
    if nm > nb:
        return False, f"print broken into fragments ({nm} small pieces vs {nb} letter bodies): tracking unreliable, horizontal bands kept"
    sizes, owner = [], {}
    for k, (ln, _, _) in enumerate(tracked):
        b, _, _ = _classify(components[np.array(ln.components)], h)
        sizes.append(len(b))
        for j in np.array(ln.components)[b]:
            owner[int(j)] = k
    total = sum(sizes)
    healthy = total > 0 and sum(s for s in sizes if s >= MIN_TRACK_BODIES) >= HEALTHY_SHARE * total
    merged = 0
    for band in band_lines:
        hits = {}
        for j in band.components:
            k = owner.get(int(j))
            if k is not None and sizes[k] >= MIN_TRACK_BODIES:
                hits[k] = hits.get(k, 0) + 1
        if sum(1 for k, n in hits.items() if n >= MIN_TRACK_BODIES) >= 2:
            merged += 1
    if not healthy:
        return False, "tracking not reliable on this block (many short tracks): horizontal bands kept"
    if not merged:
        return False, "horizontal bands agree with the tracked lines: horizontal bands kept"
    return True, f"{merged} horizontal band(s) held two or more tracked lines: tracked lines used"


def levelled_components(components: np.ndarray, tracked: list) -> np.ndarray:
    """A copy of the component boxes in which every tracked line's members are shifted vertically onto a horizontal line."""
    lev = components.copy()
    for ln, s, xr in tracked:
        for i in ln.components:
            c = components[i]
            dy = int(round(s * ((c[0] + c[2]) / 2.0 - xr)))
            lev[i, 1] = c[1] - dy
            lev[i, 3] = c[3] - dy
    return lev


def levelled_line(lev: np.ndarray, ln: Line) -> Line:
    c = lev[np.array(ln.components)]
    return Line(int(c[:, 1].min()), int(c[:, 3].max()), ln.components, ln.forced_split)


def words_of_tracked(components: np.ndarray, lev: np.ndarray, ln: Line, params, pooled) -> tuple:
    """find_words on the levelled line; word boxes rebuilt from the members' ORIGINAL boxes."""
    ws, d = find_words(lev, levelled_line(lev, ln), params, pooled)
    out = []
    for w in ws:
        c = components[np.array(w.components)]
        out.append(Word(int(c[:, 0].min()), int(c[:, 1].min()), int(c[:, 2].max()), int(c[:, 3].max()), w.components, w.kind, w.gap_before))
    return out, d


def pooled_threshold(lev: np.ndarray, tracked_lines: list, params):
    return page_threshold(lev, [levelled_line(lev, ln) for ln in tracked_lines], params)
