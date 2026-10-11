"""
Box geometry for comparing two sets of word boxes on one page: reference boxes against detected boxes.

Boxes are (x1, y1, x2, y2) integers in original page pixels, x2 and y2 exclusive (the convention of the development
annotations; a pageocr.boxes.Box converts with as_xyxy). Pure Python and deterministic: the same inputs give the same
matching, whatever the order of evaluation. No image, no recognizer and no annotation file is read here.

WHAT THE REFERENCE MUST BE. These functions compare; they do not know where the reference came from. A caller that
reports the result as agreement with human-verified boxes must pass human-verified boxes. AI-generated boxes are not
a human reference and must not be passed off as one.

SPARSE REFERENCE. The human-verified set covers only some words of a page. A detected box that touches no reference
box is then NOT a false detection: nobody judged that place. compare(..., sparse_reference=True) therefore judges only
detections that overlap a reference box, and says so in its result ("precision_scope").
"""
from dataclasses import dataclass

IOU_MATCH = 0.5          # a reference and a detection match one-to-one at or above this IoU
PART_OF = 0.5            # a box is "inside" another when at least this share of its own area lies in the other
DUPLICATE_IOU = 0.8      # two detections above this IoU with each other are duplicates of one another


class GeometryError(ValueError):
    """A box is not four integers with x1 < x2 and y1 < y2."""


def check_box(b) -> tuple:
    try:
        x1, y1, x2, y2 = (int(v) for v in b)
    except Exception:
        raise GeometryError(f"not a box: {b!r}") from None
    if any(isinstance(v, bool) or int(v) != v for v in b):
        raise GeometryError(f"box coordinates must be integers: {b!r}")
    if not (x1 < x2 and y1 < y2):
        raise GeometryError(f"empty or inverted box: {b!r}")
    return (x1, y1, x2, y2)


def as_xyxy(box) -> tuple:
    """A pageocr.boxes.Box (x, y, w, h) as (x1, y1, x2, y2)."""
    return (box.x, box.y, box.x + box.w, box.y + box.h)


def area(b) -> int:
    return (b[2] - b[0]) * (b[3] - b[1])


def intersection(a, b) -> int:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0


def iou(a, b) -> float:
    i = intersection(a, b)
    return i / (area(a) + area(b) - i) if i else 0.0


def clip_box(b, width: int, height: int):
    """The part of the box inside a width x height page, or None if nothing of it is inside."""
    x1, y1, x2, y2 = max(0, b[0]), max(0, b[1]), min(width, b[2]), min(height, b[3])
    return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None


def expand_box(b, k: int, width: int, height: int) -> dict:
    """The box with a margin of k pixels on every edge (k < 0 shrinks it), kept inside the page.

    Returns {"box": (x1, y1, x2, y2) or None, "valid": bool, "clipped_edges": [...], "requested": (...)}.
    The box is invalid (and None) if shrinking leaves less than 2 x 2 pixels. An edge that would leave the page is
    moved to the page edge and named in clipped_edges, so a margin that could not be applied in full is on record.
    """
    b = check_box(b)
    if isinstance(k, bool) or int(k) != k:
        raise GeometryError(f"the margin must be an integer number of pixels: {k!r}")
    k = int(k)
    req = (b[0] - k, b[1] - k, b[2] + k, b[3] + k)
    got = (max(0, req[0]), max(0, req[1]), min(width, req[2]), min(height, req[3]))
    clipped = [n for n, r, g in zip(("left", "top", "right", "bottom"), req, got) if r != g]
    valid = got[2] - got[0] >= 2 and got[3] - got[1] >= 2
    return {"box": got if valid else None, "valid": valid, "clipped_edges": clipped, "requested": req, "margin_px": k}


def match_one_to_one(reference: list, detected: list, threshold: float = IOU_MATCH) -> list:
    """Greedy one-to-one matching by IoU. Returns [(reference index, detection index, iou)], best pairs first.

    Deterministic: candidate pairs are taken in the order (higher IoU, lower reference index, lower detection index);
    a pair is accepted if neither box is taken yet. Each box is used at most once.
    """
    ref, det = [check_box(b) for b in reference], [check_box(b) for b in detected]
    pairs = []
    for i, r in enumerate(ref):
        for j, d in enumerate(det):
            v = iou(r, d)
            if v >= threshold:
                pairs.append((-v, i, j))
    pairs.sort()
    used_r, used_d, out = set(), set(), []
    for neg, i, j in pairs:
        if i not in used_r and j not in used_d:
            used_r.add(i); used_d.add(j)
            out.append((i, j, -neg))
    return out


@dataclass(frozen=True)
class Comparison:
    matches: tuple              # (reference index, detection index, iou), one-to-one
    missed: tuple               # reference indices with no detection on them at all
    split: tuple                # (reference index, (detection indices)): one reference word cut into several detections
    merged: tuple               # (detection index, (reference indices)): several reference words inside one detection
    poorly_localised: tuple     # reference indices touched by a detection, but not matched, split or merged
    duplicates: tuple           # (detection index, detection index) pairs lying on top of one another
    unmatched_detections: tuple  # judged detections that match no reference and are part of no split or merge
    unjudged_detections: tuple  # detections that touch no reference box (only with sparse_reference)
    counts: dict
    precision: float
    recall: float
    f1: float
    mean_iou_of_matches: float
    precision_scope: str


def compare(reference: list, detected: list, threshold: float = IOU_MATCH, sparse_reference: bool = False) -> Comparison:
    """One page: reference boxes against detected boxes.

    recall    = matched reference boxes / reference boxes
    precision = matched detections / judged detections
    A detection is judged if the reference is complete, or, with sparse_reference, if it overlaps a reference box.
    Split, merged, missed and poorly localised partition the unmatched reference boxes (a box is counted once, in
    that order of precedence: merged, split, missed, poorly localised).
    """
    ref, det = [check_box(b) for b in reference], [check_box(b) for b in detected]
    matches = match_one_to_one(ref, det, threshold)
    m_ref, m_det = {i for i, _, _ in matches}, {j for _, j, _ in matches}
    inter = {(i, j): intersection(r, d) for i, r in enumerate(ref) for j, d in enumerate(det)}
    inter = {k: v for k, v in inter.items() if v}
    # a detection that swallows two or more reference words: under-segmentation
    merged = []
    for j, d in enumerate(det):
        inside = tuple(i for i in range(len(ref)) if inter.get((i, j), 0) >= PART_OF * area(ref[i]))
        if len(inside) >= 2:
            merged.append((j, inside))
    merged_ref = {i for _, group in merged for i in group}
    # a reference word covered by two or more detections that each lie mostly inside it: over-segmentation
    split = []
    for i, r in enumerate(ref):
        parts = tuple(j for j in range(len(det)) if inter.get((i, j), 0) >= PART_OF * area(det[j]))
        if len(parts) >= 2 and i not in merged_ref:
            split.append((i, parts))
    split_ref = {i for i, _ in split}
    touched = {i for (i, _) in inter}
    unmatched_ref = [i for i in range(len(ref)) if i not in m_ref]
    in_merge = [i for i in unmatched_ref if i in merged_ref]
    in_split = [i for i in unmatched_ref if i in split_ref and i not in merged_ref]
    missed = [i for i in unmatched_ref if i not in touched]
    poor = [i for i in unmatched_ref if i in touched and i not in merged_ref and i not in split_ref]
    judged = [j for j in range(len(det)) if not sparse_reference or any((i, j) in inter for i in range(len(ref)))]
    unjudged = [j for j in range(len(det)) if j not in set(judged)]
    structural = {j for j, _ in merged} | {j for _, parts in split for j in parts}
    unmatched_det = [j for j in judged if j not in m_det and j not in structural]
    dups = tuple((a, b) for a in range(len(det)) for b in range(a + 1, len(det)) if iou(det[a], det[b]) >= DUPLICATE_IOU)
    n_match = len(matches)
    precision = n_match / len(judged) if judged else 0.0
    recall = n_match / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    counts = {"reference": len(ref), "detected": len(det), "judged_detections": len(judged), "matched": n_match,
              "reference_in_a_merge (under-segmentation)": len(in_merge), "reference_split (over-segmentation)": len(in_split),
              "reference_missed": len(missed), "reference_poorly_localised": len(poor),
              "merging_detections": len(merged), "split_events": len(split), "duplicate_pairs": len(dups),
              "unmatched_detections": len(unmatched_det), "unjudged_detections": len(unjudged)}
    assert n_match + len(in_merge) + len(in_split) + len(missed) + len(poor) == len(ref)
    return Comparison(
        matches=tuple(matches), missed=tuple(missed), split=tuple(split), merged=tuple(merged), poorly_localised=tuple(poor), duplicates=dups,
        unmatched_detections=tuple(unmatched_det), unjudged_detections=tuple(unjudged), counts=counts,
        precision=precision, recall=recall, f1=f1,
        mean_iou_of_matches=sum(v for _, _, v in matches) / n_match if n_match else 0.0,
        precision_scope=("only detections that overlap a reference box are judged; the reference does not cover the whole page"
                         if sparse_reference else "every detection is judged; the reference is taken to cover the whole page"))


def pool(comparisons: list) -> dict:
    """Counts summed over pages, with precision, recall and F1 recomputed from the sums (micro-average)."""
    keys = sorted({k for c in comparisons for k in c.counts})
    total = {k: sum(c.counts.get(k, 0) for c in comparisons) for k in keys}
    p = total["matched"] / total["judged_detections"] if total.get("judged_detections") else 0.0
    r = total["matched"] / total["reference"] if total.get("reference") else 0.0
    ious = [v for c in comparisons for _, _, v in c.matches]
    return {"counts": total, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "mean_iou_of_matches": sum(ious) / len(ious) if ious else 0.0, "pages": len(comparisons)}
