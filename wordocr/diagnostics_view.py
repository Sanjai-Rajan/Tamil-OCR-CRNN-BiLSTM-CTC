"""
The diagnostic panel of the Full-page OCR tab (app.py). Shown only when the user switches diagnostic mode on; it reads
the run's Segmentation and PageResult and never changes them. Presentation only: the analysis is wordocr.diagnostics,
the A/B crop comparison is wordocr.crop_compare.
"""
import json
import time

import streamlit as st

from pageocr import ROOT
from wordocr import crop_compare
from wordocr import diagnostics as dg

RUNS_DIR = ROOT / "diagnostic_runs"          # local evidence folder of this workspace; not a training or evaluation set
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _host_of(value: str) -> str:
    """'localhost:8501' / 'http://[::1]:8501' -> 'localhost' / '::1'."""
    value = value.split("://", 1)[-1].split("/", 1)[0]
    if value.startswith("["):
        return value[1:].split("]", 1)[0]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def opened_on_this_computer() -> bool:
    """True only when the browser reached the app through a loopback address (both the Host and the Origin header).
    Saving a bundle writes the upload to the disk of the machine running the app, so on a hosted deployment, or when
    the headers cannot be read, the save button is not offered; the ZIP download is always available."""
    try:
        headers = st.context.headers
        host, origin = headers.get("Host", ""), headers.get("Origin", "")
    except Exception:
        return False
    return bool(host and origin) and {_host_of(host), _host_of(origin)} <= LOOPBACK_HOSTS


def render(model, decoded, input_bytes: bytes, seg, result, settings) -> None:
    diag = dg.analyse(decoded, seg, result, settings)
    st.subheader("Diagnostics")
    st.caption("Read-only view of this run. Warnings are heuristics that point at regions worth a look; they are not "
               "verdicts, and no accuracy is implied (no reference text is known for an uploaded page).")
    c = diag["warning_counts"]
    cols = st.columns(4)
    cols[0].metric("Regions", diag["counts"]["regions"])
    cols[1].metric("With a warning", diag["regions_with_any_warning"])
    cols[2].metric("Lines", len(diag["lines"]))
    cols[3].metric("Empty / failed", diag["counts"]["empty"] + diag["counts"]["failed"])
    if any(c.values()):
        st.markdown("**Warnings:** " + " · ".join(f"`{k}` {n}" for k, n in c.items() if n))
    with st.expander("What each warning means"):
        for k, v in dg.WARNING_RULES.items():
            st.markdown(f"- `{k}`: {v}")

    geo = diag.get("line_geometry") or {}
    has_geo = "lines" in geo
    modes = {"Angle-aware line boxes": "angle", "Horizontal line bands (as segmented)": "horizontal"}
    choice = st.radio("Line boxes", list(modes) if has_geo else list(modes)[1:], horizontal=True, key="diag_line_mode",
                      help="Angle-aware: each line's direction estimated from its own letters (display only; the segmentation "
                           "and the crops are not changed). Horizontal: the bands the segmentation actually used.")
    mode = modes[choice]
    if not has_geo:
        st.caption("Angle-aware line boxes are unavailable for this run: " + str(geo.get("error", "no line geometry")))
    if mode == "angle":
        s, gl = geo["summary"], geo["global"]
        st.caption(f"Global deskew: measured {gl['measured_deg']:+.2f}° ({'applied' if gl['applied'] else 'not applied'}). "
                   f"Lines: {s['local']} with a local angle, {s['curved']} curved (polyline), {s['uncertain'] - s['curved']} uncertain "
                   f"(angle inherited from the global deskew, shown in magenta and marked '?'). Angles: positive = rising to the right. "
                   "Display only: word regions, crops and recognition are unchanged.")
    line_note = {"angle": "green / blue = angle-aware line boxes (alternating), magenta = no reliable local angle",
                 "horizontal": "green = line bands"}[mode]
    t1, t2 = st.tabs(["Original page, region numbers", "Crop frame, lines and regions"])
    t1.image(dg.overlay_original(decoded, diag, seg.h, mode, seg), width="stretch",
             caption=f"Boxes in the coordinates of the uploaded image; numbers as in the table. Orange = at least one warning. {line_note}.")
    t2.image(dg.overlay_ocr(seg, diag, True, mode), width="stretch",
             caption="The image the crops are cut from (" + diag["frames"]["ocr"] + f"); {line_note}.")

    st.markdown("**Regions** (click a column header to sort)")
    only = st.checkbox("Only regions with a warning", key="diag_only_warned")
    rows = [r for r in dg.region_rows(diag) if (r["warnings"] or not only)]
    st.dataframe(rows, hide_index=True, width="stretch", height=320)

    st.markdown("**Inspect one region**")
    n = st.number_input("Region number", min_value=1, max_value=max(1, len(diag["regions"])), value=1, step=1, key="diag_region")
    if diag["regions"]:
        r = diag["regions"][int(n) - 1]
        crop = dg.pipeline.ocr_crops(seg.ocr, [seg.words[int(n) - 1]], diag["margin_px"])[0]
        i1, i2 = st.columns([1, 2])
        i1.image(crop, caption=f"#{r['number']} {r['id']}: the crop the model read ({r['crop_width']} x {r['crop_height']} px, actual size)")
        i2.image(dg.crop_context(seg, diag, int(n)), caption="In context: red = tight box, blue = crop with margin", width="stretch")
        st.json({k: r[k] for k in ("id", "status", "text", "warnings", "box_original", "box_ocr", "crop_box_ocr", "gap_before_px", "uncalibrated_score", "error")})

    with st.expander(f"Lines ({len(diag['lines'])})"):
        st.dataframe([{k: v for k, v in ln.items() if k != "regions"} | {"regions": len(ln["regions"])} for ln in diag["lines"]], hide_index=True, width="stretch")
        if has_geo:
            st.markdown("**Line geometry** (angle-aware; display only)")
            st.dataframe([{"id": g["id"], "angle_original_deg": g["angle_original_deg"], "angle_ocr_deg": g["angle_ocr_deg"], "source": g["source"],
                           "reliable": g["reliable"], "shape": g["shape"], "bodies": g["bodies"], "marks": g["marks"], "rule_like": g["rule_like"],
                           "spread_deg": g["fit"]["spread_deg"], "residual_h": g["fit"]["median_residual_h"],
                           "sag_h": (g.get("curvature") or {}).get("sag_h"), "band_crosses_other_lines": g["other_lines_components_in_band"],
                           "overlap_with_next_px": g["overlap_with_next_px"], "warnings": "; ".join(g["warnings"])} for g in geo["lines"]],
                         hide_index=True, width="stretch")

    st.markdown("**Evidence bundle**")
    inc = st.checkbox("Include the input image in the bundle", value=True, key="diag_include_input")
    b1, b2 = st.columns(2)
    b1.download_button("Download diagnostic bundle (ZIP)", dg.bundle_zip(decoded, seg, result, settings, diag, inc, input_bytes),
                       file_name=f"diagnostic_{decoded.sha256[:12]}.zip", mime="application/zip", on_click="ignore", key="diag_zip", width="stretch")
    if not opened_on_this_computer():
        b2.caption("Saving to the app's disk is offered only when the app is opened on the computer that runs it.")
    elif b2.button("Save bundle to diagnostic_runs/", key="diag_save", width="stretch"):
        folder = RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{decoded.sha256[:12]}"
        try:
            dg.save_bundle(folder, decoded, seg, result, settings, diag, inc, input_bytes)
            st.success(f"Saved to diagnostic_runs/{folder.name}")
        except Exception as e:
            st.error(f"Not saved: {e}")

    st.markdown("**Manual crop comparison** (condition A: your box; condition B: the pipeline's crop; same model)")
    st.caption("Enter boxes in the coordinates of the uploaded image (see the first overlay). Enter the reference text only "
               "after checking it yourself, and tick 'confirmed'; unconfirmed references are never scored. The model's "
               "output is never used as a reference.")
    template = [{"id": "S01", "x1": 0, "y1": 0, "x2": 0, "y2": 0, "reference": "", "confirmed": False}]
    table = st.data_editor(st.session_state.get("diag_samples", template), num_rows="dynamic", key="diag_samples_editor", width="stretch")
    if st.button("Compare A and B", key="diag_compare"):
        samples = [crop_compare.Sample(id=str(s["id"]), box=(int(s["x1"]), int(s["y1"]), int(s["x2"]), int(s["y2"])),
                                       reference=str(s.get("reference") or ""), reference_confirmed=bool(s.get("confirmed")))
                   for s in table if s.get("id")]
        rep = crop_compare.compare(model, decoded, seg, diag["margin_px"], samples, result=result)
        st.session_state["diag_compare_result"] = {"input_sha256": decoded.sha256, "report": rep}
    got = st.session_state.get("diag_compare_result")
    if got and got["input_sha256"] == decoded.sha256:
        rep = got["report"]
        st.json(rep["summary"])
        st.dataframe([{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in s.items()} for s in rep["samples"]],
                     hide_index=True, width="stretch")
        st.download_button("Download comparison (JSON)", json.dumps(rep, indent=1, ensure_ascii=False).encode("utf-8"),
                           file_name=f"crop_comparison_{decoded.sha256[:12]}.json", mime="application/json", on_click="ignore", key="diag_cmp_json")
