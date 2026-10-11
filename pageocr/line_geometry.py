"""
Angle-aware geometry of the text lines of a segmentation: for viewing and diagnosis only.

    geo = line_geometry(seg)           # one record per line, in the order of seg.lines (block by block, top to bottom)

The segmentation finds lines as HORIZONTAL bands of rows of the layout image (pageocr.lines). This module does not
change that. It describes each found line by the direction of its own ink:

  1. Members. The line's connected components (the same ones the segmentation assigned to it; the layout image is
     rebuilt read-only from seg.ocr and seg.params, exactly as the pipeline built it last) are sorted into
       - rule-like: very long and flat (notebook ruling, underline, a rule fused with letters); never used for the angle,
       - marks: small (pulli, dots, vowel-sign fragments); kept inside the box, not used for the angle,
       - bodies: the rest; their box centres are the points of the fit.
  2. Direction. Theil-Sen estimate: the median slope over all pairs of body points at least 2 x text height apart,
     intercept = median residual. Robust to a minority of stray points (isolated marks, a mis-assigned component).
  3. Reliability. The local angle is reported only if there are >= 4 bodies spanning >= 4 x text height, the half
     inter-quartile range of the pair angles is <= 1.0 degree, and the median absolute residual is <= 0.3 x text
     height. Otherwise the line is "uncertain": the angle shown is the one inherited from the global deskew (0 in the
     OCR frame), labelled as inherited, and no local angle is claimed.
  4. Curvature. A robust quadratic through the body points: if it bows by more than 0.35 x text height over the line
     AND the bow is at least 4 standard errors (least-squares covariance, noise from the MAD), or if the two halves differ
     in angle by > 1.5 degrees, the line is flagged "curved or changing angle" and its box is a polygon around a centre
     polyline (the quadratic sampled at 9 points) instead of one straight box. Tamil letters differ in height, so the
     centre of a single letter wanders by about 0.15 x text height; the thresholds are set well above that.
  5. Box. Straight lines: the rotated rectangle that holds every non-rule member component, in the line's direction
     (projections of the component corners onto the direction and its normal). Curved lines: the centre polyline
     shifted up and down (vertically) by the extent of the member components.
  6. Frames. Everything is computed in the OCR frame (the image crops are cut from, rotated if deskewed) and mapped to
     the ORIGINAL uploaded image with the OCR image's exact affine (OcrImage.to_original). Angles are visual: positive =
     the line rises to the right (counter-clockwise); angle_original = the direction mapped into the original frame.
  7. Neighbours. For each line: how much of its box area overlaps the next line's box in the same block, and how many
     components of OTHER lines its horizontal band crosses (the evidence for "is the horizontal band only a display
     limitation, or does it cut through neighbouring text?").

Nothing here changes a box, a crop, a word or a recognizer input.
"""
import math

import cv2
import numpy as np

from pageocr.layout import build_layout_image

GEOMETRY_VERSION = "pageocr-line-geometry-1.0.0"
MIN_BODIES = 4
MIN_SPAN_H = 4.0
MAX_SPREAD_DEG = 1.0
MAX_RESID_H = 0.3
CURVE_SAG_H = 0.35
HALVES_DIFF_DEG = 1.5


def _classify(comps: np.ndarray, h: float):
    """(bodies, marks, rules) as index arrays into comps."""
    w = comps[:, 2] - comps[:, 0]
    ht = comps[:, 3] - comps[:, 1]
    rule = ((w > 6 * h) & (ht < 0.6 * h)) | (w > 12 * np.maximum(ht, 1))
    mark = ~rule & ((ht < 0.4 * h) | (np.maximum(w, ht) < 0.45 * h))
    body = ~rule & ~mark
    return np.nonzero(body)[0], np.nonzero(mark)[0], np.nonzero(rule)[0]


def _theil_sen(x: np.ndarray, y: np.ndarray, min_dx: float):
    i, j = np.triu_indices(len(x), 1)
    dx, dy = x[j] - x[i], y[j] - y[i]
    keep = np.abs(dx) >= min_dx
    if keep.sum() < 3:
        return None
    s = dy[keep] / dx[keep]
    slope = float(np.median(s))
    ang = np.degrees(np.arctan(s))
    spread = float((np.percentile(ang, 75) - np.percentile(ang, 25)) / 2.0)
    return slope, float(np.median(y - slope * x)), spread


def _to_original(ocr, pts: np.ndarray) -> np.ndarray:
    a = np.array(ocr.to_original, float)
    return np.c_[pts, np.ones(len(pts))] @ a.T


def _visual_angle(p0, p1) -> float:
    """Angle of the direction p0 -> p1 in image coordinates (y down), positive when rising to the right."""
    return float(-math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])))


def _rect(corners_xy: np.ndarray, theta: float) -> np.ndarray:
    """The rotated rectangle in direction theta (radians, image coords) holding all points: 4 x 2, clockwise from top-left."""
    u = np.array([math.cos(theta), math.sin(theta)])
    n = np.array([-math.sin(theta), math.cos(theta)])          # points "down" for small angles
    a, b = corners_xy @ u, corners_xy @ n
    a1, a2, b1, b2 = a.min(), a.max(), b.min(), b.max()
    return np.array([a1 * u + b1 * n, a2 * u + b1 * n, a2 * u + b2 * n, a1 * u + b2 * n])


def _corners(c: np.ndarray) -> np.ndarray:
    return np.array([[c[0], c[1]], [c[2], c[1]], [c[2], c[3]], [c[0], c[3]]], float)


def _poly_area_overlap(p: np.ndarray, q: np.ndarray, shape) -> int:
    """Pixels covered by both polygons (rasterised in the OCR frame)."""
    H, W = shape
    x1 = int(max(0, math.floor(min(p[:, 0].min(), q[:, 0].min()))))
    y1 = int(max(0, math.floor(min(p[:, 1].min(), q[:, 1].min()))))
    x2 = int(min(W, math.ceil(max(p[:, 0].max(), q[:, 0].max())) + 1))
    y2 = int(min(H, math.ceil(max(p[:, 1].max(), q[:, 1].max())) + 1))
    if x2 <= x1 or y2 <= y1:
        return 0
    a = np.zeros((y2 - y1, x2 - x1), np.uint8)
    b = np.zeros_like(a)
    off = np.array([x1, y1], float)
    cv2.fillPoly(a, [np.round(p - off).astype(np.int32)], 1)
    cv2.fillPoly(b, [np.round(q - off).astype(np.int32)], 1)
    return int((a & b).sum())


def rebuild_components(seg) -> np.ndarray:
    """The component array the segmentation's lines index into (read-only rebuild of the last layout pass)."""
    layout = build_layout_image(seg.ocr, seg.params)
    n = max((max(ln.components) for ls in seg.lines for ln in ls if ln.components), default=-1)
    if n >= len(layout.components) or abs(float(layout.h) - float(seg.h)) > 1e-6:
        raise RuntimeError("the rebuilt layout image does not match this segmentation")
    for b, ls in zip(seg.blocks, seg.lines):                 # every member must overlap its own band, as find_lines assigned it
        for ln in ls:
            c = layout.components[np.array(ln.components, int)]
            if ((np.minimum(c[:, 3], ln.y2) - np.maximum(c[:, 1], ln.y1)) <= 0).any() or (c[:, 2] <= b.x1).any() or (c[:, 0] >= b.x2).any():
                raise RuntimeError("the rebuilt layout image does not match this segmentation")
    return layout.components


def _line_record(seg, comps, block, bi, li, ln, h, global_ocr_angle=0.0):
    rec = {"block": bi, "line": li, "id": f"b{bi:02d}_l{li:03d}", "band_ocr": [int(block.x1), int(ln.y1), int(block.x2), int(ln.y2)],
           "warnings": []}
    idx = np.array(ln.components, int)
    mine = comps[idx]
    bodies, marks, rules = _classify(mine, h)
    rec.update(components=len(idx), bodies=len(bodies), marks=len(marks), rule_like=len(rules))
    if len(rules):
        rec["warnings"].append("rule-like components in the line (ruling or underline): ignored for the angle")
    pts = np.c_[(mine[bodies, 0] + mine[bodies, 2]) / 2.0, (mine[bodies, 1] + mine[bodies, 3]) / 2.0] if len(bodies) else np.zeros((0, 2))
    span = float(pts[:, 0].max() - pts[:, 0].min()) if len(pts) else 0.0
    fit = _theil_sen(pts[:, 0], pts[:, 1], 2.0 * h) if len(pts) >= MIN_BODIES and span >= MIN_SPAN_H * h else None
    reliable, curved, slope, icpt, spread, resid = False, False, None, None, None, None
    if fit is not None:
        slope, icpt, spread = fit
        res = pts[:, 1] - (slope * pts[:, 0] + icpt)
        resid = float(np.median(np.abs(res)))
        reliable = spread <= MAX_SPREAD_DEG and resid <= MAX_RESID_H * h
        # curvature: a robust quadratic (least squares, refitted once without points beyond 2.5 MAD) must both bow by
        # more than CURVE_SAG_H x text height over the line and fit clearly better than the straight line; or the two
        # halves of the line must differ in angle by more than HALVES_DIFF_DEG
        order = np.argsort(pts[:, 0])
        quad, sag, sag_se = None, 0.0, float("inf")
        if len(pts) >= 2 * MIN_BODIES:
            xm = float(pts[:, 0].mean())
            q = np.polyfit(pts[:, 0] - xm, pts[:, 1], 2)
            r2 = pts[:, 1] - np.polyval(q, pts[:, 0] - xm)
            mad = float(np.median(np.abs(r2))) or 1e-6
            ok = np.abs(r2) <= 2.5 * 1.4826 * mad
            if ok.sum() < 2 * MIN_BODIES:
                ok = np.ones(len(pts), bool)
            X = np.vander(pts[ok, 0] - xm, 3)
            q = np.linalg.lstsq(X, pts[ok, 1], rcond=None)[0]
            sigma = 1.4826 * float(np.median(np.abs(pts[ok, 1] - X @ q))) or 1e-6
            cov = np.linalg.pinv(X.T @ X) * sigma ** 2
            quad = (q, xm)
            sag = abs(float(q[0])) * (span / 2.0) ** 2
            sag_se = math.sqrt(max(float(cov[0, 0]), 0.0)) * (span / 2.0) ** 2
        half = len(order) // 2
        halves = [_theil_sen(pts[order[s], 0], pts[order[s], 1], 2.0 * h) for s in (slice(0, half), slice(half, None))] if half >= MIN_BODIES else [None, None]
        hd = (abs(math.degrees(math.atan(halves[0][0])) - math.degrees(math.atan(halves[1][0])))
              if halves[0] and halves[1] and max(halves[0][2], halves[1][2]) <= MAX_SPREAD_DEG else 0.0)
        bowed = quad is not None and sag > CURVE_SAG_H * h and sag > 4.0 * sag_se
        if bowed or hd > HALVES_DIFF_DEG:
            curved = True
            reliable = False
            rec["warnings"].append("curved or changing angle along the line: drawn as a polyline polygon, no single angle")
            xs = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 9)
            if quad is not None:
                centre = np.c_[xs, np.polyval(quad[0], xs - quad[1])]
            else:                                             # angle change without a usable quadratic: two straight halves
                centre = np.c_[xs, np.where(xs < pts[order[half], 0], halves[0][0] * xs + halves[0][1], halves[1][0] * xs + halves[1][1])]
        rec["curvature"] = {"sag_h": round(sag / max(h, 1), 3), "sag_se_h": round(sag_se / max(h, 1), 3) if math.isfinite(sag_se) else None,
                            "halves_angle_diff_deg": round(hd, 2)}
    else:
        rec["warnings"].append(f"too few letter bodies for a local angle ({len(bodies)} bodies, span {span / max(h, 1):.1f} x text height)")
    if fit is not None and not reliable and not curved:
        rec["warnings"].append(f"local angle uncertain (spread {spread:.2f} deg, residual {resid / max(h, 1):.2f} x text height)")
    keep = np.r_[bodies, marks].astype(int) if len(bodies) + len(marks) else np.arange(len(mine))
    corners = np.vstack([_corners(c) for c in mine[keep]])
    if curved:
        x0, x1 = float(mine[keep, 0].min()), float(mine[keep, 2].max())
        cx = np.r_[x0, centre[:, 0], x1]
        cy = np.r_[centre[0, 1], centre[:, 1], centre[-1, 1]]
        off_top = min(float(c[1]) - np.interp((c[0] + c[2]) / 2.0, cx, cy) for c in mine[keep])
        off_bot = max(float(c[3]) - np.interp((c[0] + c[2]) / 2.0, cx, cy) for c in mine[keep])
        poly = np.r_[np.c_[cx, cy + off_top], np.c_[cx[::-1], cy[::-1] + off_bot]]
        rec.update(shape="polyline", source="local polyline (curved)")
        theta_ocr = None
    else:
        if reliable:
            theta = math.atan(slope)
            rec["source"] = "local"
        else:
            theta = math.radians(-global_ocr_angle)
            rec["source"] = "inherited from global deskew" if seg.skew.get("applied") else "none: horizontal assumed (no deskew applied)"
        poly = _rect(corners, theta)
        rec["shape"] = "rotated_rectangle"
        theta_ocr = theta
    rec["polygon_ocr"] = [[round(float(x), 1), round(float(y), 1)] for x, y in poly]
    po = _to_original(seg.ocr, poly)
    rec["polygon_original"] = [[round(float(x), 1), round(float(y), 1)] for x, y in po]
    rec["reliable"] = bool(reliable)
    if theta_ocr is not None:
        p0 = np.array([[0.0, 0.0], [1000.0 * math.cos(theta_ocr), 1000.0 * math.sin(theta_ocr)]])
        q = _to_original(seg.ocr, p0)
        rec["angle_ocr_deg"] = round(_visual_angle(p0[0], p0[1]), 2)
        rec["angle_original_deg"] = round(_visual_angle(q[0], q[1]), 2)
    else:
        rec["angle_ocr_deg"] = rec["angle_original_deg"] = None
    rec["fit"] = {"points": int(len(pts)), "span_h": round(span / max(h, 1), 2), "spread_deg": None if spread is None else round(spread, 3),
                  "median_residual_h": None if resid is None else round(resid / max(h, 1), 3)}
    # the horizontal band: how many components of OTHER lines of the block it crosses, and own components it cuts
    return rec, idx


def line_geometry(seg, comps: np.ndarray | None = None) -> dict:
    """Angle-aware description of every line of `seg` (see the module docstring)."""
    comps = rebuild_components(seg) if comps is None else comps
    h = float(seg.h) or 20.0
    global_ocr = 0.0                       # after deskew the OCR frame is levelled; without deskew nothing was applied
    out, owner = [], {}
    for bi, (b, ls) in enumerate(zip(seg.blocks, seg.lines)):
        for li, ln in enumerate(ls):
            for c in ln.components:
                owner[int(c)] = (bi, li)
    for bi, (b, ls) in enumerate(zip(seg.blocks, seg.lines)):
        for li, ln in enumerate(ls):
            rec, idx = _line_record(seg, comps, b, bi, li, ln, h, global_ocr)
            mine = comps[idx]
            rec["own_components_cut_by_band"] = int(((mine[:, 1] < ln.y1) | (mine[:, 3] > ln.y2)).sum())
            other = [i for i, c in enumerate(comps) if owner.get(i) not in (None, (bi, li)) and c[1] < ln.y2 and c[3] > ln.y1
                     and c[0] < b.x2 and c[2] > b.x1]
            rec["other_lines_components_in_band"] = len(other)
            out.append(rec)
    by_block = {}
    for r in out:
        by_block.setdefault(r["block"], []).append(r)
    for rs in by_block.values():
        for a, b in zip(rs, rs[1:]):
            ov = _poly_area_overlap(np.array(a["polygon_ocr"]), np.array(b["polygon_ocr"]), seg.ocr.gray.shape)
            area = max(1.0, float(cv2.contourArea(np.array(a["polygon_ocr"], np.float32))))
            a["overlap_with_next_px"] = ov
            if ov > 0.05 * area:
                a["warnings"].append(f"its box overlaps the next line's box by {ov} px ({100 * ov / area:.0f} % of its area): the lines touch or interleave")
        rs[-1]["overlap_with_next_px"] = 0
    deskew = seg.skew
    return {"version": GEOMETRY_VERSION, "text_height_px": round(h, 2),
            "global": {"measured_deg": deskew.get("angle_deg"), "applied": bool(deskew.get("applied")),
                       "ocr_frame": "levelled by the global deskew" if deskew.get("applied") else "the original frame (no deskew applied)",
                       "angle_convention": "visual degrees, positive = the line rises to the right"},
            "summary": {"lines": len(out), "local": sum(r["source"] == "local" for r in out),
                        "curved": sum(r["shape"] == "polyline" for r in out), "uncertain": sum(not r["reliable"] for r in out),
                        "lines_whose_band_crosses_other_lines": sum(r["other_lines_components_in_band"] > 0 for r in out)},
            "lines": out}


COLOURS = ((0, 170, 0), (0, 150, 210))      # alternate line colours (RGB): neighbouring lines stay distinguishable
UNCERTAIN = (200, 0, 200)                    # magenta: no reliable local angle (word regions use red / orange)


def draw_lines(img_rgb: np.ndarray, geo: dict, frame: str = "ocr", mode: str = "angle", seg=None, label: bool = True) -> np.ndarray:
    """Draw the lines on an RGB image of the given frame. mode "angle": the angle-aware boxes; "horizontal": the
    segmentation's horizontal bands (mapped to the original frame if needed: there they are rotated quadrilaterals)."""
    if frame not in ("ocr", "original") or mode not in ("angle", "horizontal"):
        raise ValueError("frame is 'ocr' or 'original'; mode is 'angle' or 'horizontal'")
    out = img_rgb.copy()
    h = float(geo.get("text_height_px") or 20)
    th = max(1, int(round(h / 22.0)))
    scale = max(0.35, min(1.0, h / 34.0))
    for k, r in enumerate(geo["lines"]):
        if mode == "horizontal":
            x1, y1, x2, y2 = r["band_ocr"]
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)
            if frame == "original":
                if seg is None:
                    raise ValueError("the horizontal bands need seg to be mapped to the original frame")
                poly = _to_original(seg.ocr, poly)
            colour = COLOURS[k % 2]
        else:
            poly = np.array(r["polygon_ocr" if frame == "ocr" else "polygon_original"], float)
            colour = COLOURS[k % 2] if r["reliable"] else UNCERTAIN
        cv2.polylines(out, [np.round(poly).astype(np.int32)], True, colour, th, cv2.LINE_AA)
        if label:
            ang = r["angle_original_deg" if frame == "original" else "angle_ocr_deg"]
            txt = r["id"][4:] if mode == "horizontal" else (f"{r['id'][4:]} {ang:+.1f}" if r["reliable"] and ang is not None else f"{r['id'][4:]} ?")
            x, y = poly[:, 0].min(), poly[:, 1].min()
            cv2.putText(out, txt, (int(max(0, x - 4 - 60 * scale)), int(max(10, y + 12 * scale))), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, colour, 1, cv2.LINE_AA)
    return out
