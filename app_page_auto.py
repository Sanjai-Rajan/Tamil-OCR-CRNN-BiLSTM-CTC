"""
Tamil OCR - Automatic page segmentation (V3 preview). Segmentation diagnostics only, until the development guard opens.

    .venv\\Scripts\\python -m streamlit run app_page_auto.py

Upload a page image. The app finds text blocks, lines and word boxes with classical image processing
(pageocr.pipeline), shows them on the page, and exports the segmentation as JSON. The intended flow is

    upload -> layout -> lines -> word boxes -> OCR -> text in reading order -> TXT / JSON

but the OCR step is DISABLED: the recognizer is not run from this app until the human verification of the development
pages is complete and frozen (pageocr.devguard). There is no setting that enables it earlier.

Two image paths: text is FOUND on a processed layout image; any crop for the recognizer would be cut from the original
grayscale page. The segmentation has been tested on generated pages only. It is not validated full-page OCR.

Nothing is trained or written to disk; results leave only through the download buttons. app.py (V1) and app_page.py
(V2, operator-assisted) are separate entry points and are not changed by this file.
"""
import hashlib
import json

import cv2
import streamlit as st

from pageocr import ROOT, boxes as bx, devguard, imageio as pio, pipeline, reading_order
from pageocr.layout import SegParams

CHECKPOINT = ROOT / "runs" / "experiment_002a" / "best.pth"
PREVIEW_WIDTH = 1150
TITLE = "Tamil OCR — Automatic Page Segmentation (preview)"


@st.cache_resource(show_spinner=False)
def protected_index() -> dict:
    return pio.protected_size_index()


@st.cache_data(show_spinner=False)
def guard_status() -> dict:
    return devguard.status()


st.set_page_config(page_title=TITLE, layout="wide")
state = st.session_state
state.setdefault("page", None)
state.setdefault("seg", None)
state.setdefault("ocr", None)

st.title(TITLE)
st.info(pipeline.STATEMENT)
gs = guard_status()
ocr_enabled = bool(gs["ready"])
if not ocr_enabled:
    st.warning("OCR is disabled in this app. The recognizer is not run here until the human verification of the development pages is complete and frozen. "
               "Segmentation, its display and its export work now.")

with st.container(border=True):
    st.subheader("1. Page image")
    upload = st.file_uploader("Upload one page image", type=list(pio.EXTENSIONS), accept_multiple_files=False, key="page_upload")
    st.caption("PNG, JPG/JPEG or single-page TIFF, grayscale or colour, as scanned (not binarized). No PDF.")

if upload is None:
    state.page, state.seg, state.ocr = None, None, None
else:
    data = upload.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    if state.page is None or state.page.sha256 != sha:
        state.page, state.seg, state.ocr = None, None, None
        try:
            if pio.protected_match(data, protected_index()) is not None:
                raise pio.PageError("This file is part of the frozen research data (real_docs / R1) and is not accepted here.")
            state.page = pio.load_page_bytes(data, upload.name)
        except pio.PageError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"This file could not be opened ({type(e).__name__}).")
page = state.page
if page is None:
    st.stop()

if state.seg is None:
    try:
        with st.spinner("Finding blocks, lines and words ..."):
            state.seg = pipeline.segment(page, SegParams())
    except Exception as e:
        st.error(f"The page could not be segmented ({type(e).__name__}: {e}).")
        st.stop()
seg = state.seg
diag = pipeline.diagnostics(seg)

with st.container(border=True):
    st.subheader("2. Layout, lines and word boxes")
    c = diag["counts"]
    st.caption(f"original {page.width} × {page.height} px · {c['blocks']} blocks · {c['lines']} lines · {c['words']} word boxes · "
               f"text height {seg.h:.0f} px · skew {seg.skew['angle_deg']:+.2f}°" + (" (corrected)" if seg.skew["applied"] else ""))
    show = st.multiselect("Show", ["blocks", "lines", "words"], default=["blocks", "lines", "words"], key="show_layers")
    overlay = cv2.cvtColor(pipeline.draw_overlay(seg, tuple(show)), cv2.COLOR_BGR2RGB)
    width = st.slider("Zoom (display width in pixels)", 500, max(600, min(4000, overlay.shape[1] * 2)), min(PREVIEW_WIDTH, overlay.shape[1]), 50, key="zoom")
    st.image(overlay, width=width)
    st.caption("Blue: blocks, numbered in reading order. Green: line bands. Red: word boxes; orange: punctuation on its own. "
               "The picture is the OCR image (original gray" + (", deskewed" if seg.skew["applied"] else "") + "); the boxes were found on the layout image.")
    for f in seg.flags:
        st.warning(f)

with st.container(border=True):
    st.subheader("3. Export the segmentation")
    st.caption("Diagnostics only: geometry, parameters and flags. No recognizer output is in these files.")
    boxes = pipeline.boxes_for_export(seg, "original")
    a, b = st.columns(2)
    a.download_button("Download segmentation diagnostics (JSON)", json.dumps(diag, ensure_ascii=False, indent=1).encode("utf-8"),
                      file_name=f"segmentation_{page.sha256[:12]}.json", mime="application/json", key="dl_diag")
    b.download_button("Download word boxes (JSON, original page pixels)", bx.dumps_json(boxes, page.width, page.height, page.sha256).encode("utf-8"),
                      file_name=f"boxes_{page.sha256[:12]}.json", mime="application/json", key="dl_boxes", disabled=not boxes)

with st.container(border=True):
    st.subheader("4. OCR and text in reading order")
    if not ocr_enabled:
        st.button("Run OCR", disabled=True, key="run_ocr")
        st.caption("Disabled. Still missing: " + "; ".join(gs["missing"]))
    else:
        if st.button("Run OCR", type="primary", key="run_ocr", disabled=not seg.words):
            from pageocr.engine import OcrEngine
            try:
                engine = OcrEngine(str(CHECKPOINT))
                recs = pipeline.recognize(seg, engine)
                texts = [r.text if r is not None else "" for r in recs]
                state.ocr = {"texts": texts, "blocks": reading_order.as_text_lines(list(seg.words), texts), "checkpoint_sha256": engine.checkpoint_sha256}
            except Exception as e:
                st.error(f"OCR did not run ({type(e).__name__}: {e}).")
        if state.ocr:
            text = "\n\n".join("\n".join(lines) for _, lines in state.ocr["blocks"]) + "\n"
            st.text_area("Text in reading order (raw greedy output; not validated)", text, height=320)
            st.download_button("Download TXT", text.encode("utf-8"), file_name=f"text_{page.sha256[:12]}.txt", mime="text/plain", key="dl_txt")
            doc = dict(diag, recognizer={"checkpoint_sha256": state.ocr["checkpoint_sha256"], "note": "raw greedy CTC output; confidence uncalibrated"},
                       texts=[{"id": w.id, "text": t} for w, t in zip(seg.words, state.ocr["texts"])])
            st.download_button("Download JSON", json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8"), file_name=f"page_{page.sha256[:12]}.json",
                               mime="application/json", key="dl_json")
