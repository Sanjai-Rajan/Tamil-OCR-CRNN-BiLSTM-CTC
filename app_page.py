"""
Tamil OCR - Operator-Assisted Region OCR (V2). Inference only.

    .venv\\Scripts\\python -m streamlit run app_page.py

The operator uploads a page image and supplies word boxes (a JSON/CSV file, or the table on the page). Each box is cut
from the ORIGINAL grayscale page and read by the frozen Experiment 002-A word recognizer (pageocr.engine). The text is
assembled in the operator's order (column_index, line_index, box order). This app finds no text, lines, columns or
reading order by itself. It is not validated full-page OCR.

Nothing is trained, evaluated or written to disk: results leave only through the download buttons. app.py (V1) is a
separate entry point and is not used here.
"""
import base64
import hashlib
import re
from pathlib import Path

import cv2
import streamlit as st

from pageocr import ROOT, boxes as bx, devguard, export, imageio as pio
from pageocr.engine import CheckpointError, InvalidCropError, OcrEngine, validate_crop

CHECKPOINT = ROOT / "runs" / "experiment_002a" / "best.pth"
PREVIEW_WIDTH = 1100


@st.cache_resource(show_spinner="Loading the model ...")
def load_engine(path: str) -> OcrEngine:
    return OcrEngine(path)


_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\](){}<>#!|~$:])")


def md(text) -> str:
    """Untrusted text (file names, box ids, messages that quote them) made inert for Streamlit Markdown: no links,
    images, formatting or directives can be injected through it."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", str(text))


@st.cache_resource(show_spinner=False)
def protected_index() -> dict:
    return pio.protected_size_index()


def thumbnail(crop) -> str:
    """A PNG data URL of the crop for the results table (display only; the recognizer gets the array itself)."""
    ok, png = cv2.imencode(".png", crop)
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode() if ok else ""


def rows_to_boxes(rows) -> list[bx.Box]:
    """The table's rows as boxes. Whole numbers only; an incomplete or fractional cell is an error, not a guess."""
    out = []
    for n, row in enumerate(rows, start=1):
        cells = {}
        for k in bx.INT_FIELDS:
            v = row.get(k)
            if v is None or isinstance(v, bool) or v != v or int(v) != v:
                raise bx.BoxError(f"table row {n}: {k} must be a whole number, got {v!r}")
            cells[k] = int(v)
        out.append(bx.Box(id=str(row.get("id") or "").strip(), **cells))
    return out


# ---------------------------------------------------------------- page

st.set_page_config(page_title="Tamil OCR — Operator-Assisted Region OCR", layout="wide")
st.markdown("""
<style>
  html, body, [class*="css"], textarea, input { font-family: "Nirmala UI", "Latha", "Noto Sans Tamil", "Vijaya", "Segoe UI", sans-serif; }
  .block-container { padding-top: 2.2rem; max-width: 1250px; }
  textarea { font-size: 1.25rem !important; line-height: 1.6 !important; }
  .ocr-sub { color: #5f6368; font-size: 1.05rem; margin-top: -0.6rem; }
</style>
""", unsafe_allow_html=True)

state = st.session_state
state.setdefault("page", None)             # pageocr.imageio.Page of the current upload
state.setdefault("boxes", [])              # list of Box, in ORIGINAL page coordinates
state.setdefault("imported", None)         # fingerprint of the box file already applied
state.setdefault("table_round", 0)         # changing the editor's key reloads it from state.boxes
state.setdefault("result", None)           # {"key", "boxes", "crops", "recognitions", "skipped"}

st.title("Tamil OCR — Operator-Assisted Region OCR")
st.markdown('<p class="ocr-sub">Frozen Experiment 002-A word-level recognizer</p>', unsafe_allow_html=True)
st.info(export.STATEMENT + " You supply the word boxes and their order; the model reads one box at a time.")

engine = None
try:
    engine = load_engine(str(CHECKPOINT))
except CheckpointError as e:
    st.error(f"The model could not be loaded. {e}")
except Exception as e:
    st.error(f"The model could not be loaded ({type(e).__name__}).")

# ---- 1. page image
with st.container(border=True):
    st.subheader("1. Page image")
    upload = st.file_uploader("Upload one page image", type=list(pio.EXTENSIONS), accept_multiple_files=False, key="page_upload")
    st.caption("PNG, JPG/JPEG or single-page TIFF, grayscale or colour, as scanned (not binarized). No PDF.")

if upload is None:
    state.page = None
else:
    data = upload.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    if state.page is None or state.page.sha256 != sha:
        state.page, state.boxes, state.imported, state.result = None, [], None, None
        state.table_round += 1
        try:
            if pio.protected_match(data, protected_index()) is not None:
                raise pio.PageError("This file is part of the frozen research data (real_docs / R1) and is not accepted here.")
            devguard.assert_ocr_allowed(sha)        # a development page is refused until the human-verified set is frozen
            loaded = pio.load_page_bytes(data, upload.name)
            devguard.assert_ocr_allowed(devguard.development_page_for_pixels(loaded.gray))    # also as a re-encoded copy
            state.page = loaded
        except devguard.DevelopmentBlocked:
            st.error("This file is a development page. The recognizer may not be run on development pages until their "
                     "human verification is complete and frozen (pageocr/devguard.py), so it is not accepted here.")
        except pio.PageError as e:
            st.error(md(e))
        except Exception as e:
            st.error(f"This file could not be opened ({type(e).__name__}).")
page = state.page

if page is None:
    st.stop()

preview, scale = pio.make_preview(page.gray, PREVIEW_WIDTH)        # the preview is a reduced copy of the same array

# ---- 2. boxes
with st.container(border=True):
    st.subheader("2. Word boxes")
    st.caption(f"Coordinates are whole pixels of the original image ({page.width} × {page.height}): x, y = top-left corner, "
               "w, h = size. line_index and column_index give the reading order; nothing is detected automatically.")
    tab_import, tab_table = st.tabs(["Import JSON / CSV", "Edit table"])
    with tab_import:
        box_upload = st.file_uploader("Box file", type=["json", "csv"], key=f"box_upload_{page.sha256[:12]}")
        if box_upload is not None:
            raw = box_upload.getvalue()
            fingerprint = hashlib.sha256(raw).hexdigest()
            if state.imported != fingerprint:
                try:
                    text = raw.decode("utf-8-sig")
                    loaded = bx.loads_csv(text) if box_upload.name.lower().endswith(".csv") else bx.loads_json(text)
                    bx.validate_boxes(loaded.boxes, page.width, page.height)
                    made_for = (loaded.page or {}).get("sha256")
                    if made_for and made_for != page.sha256:
                        raise bx.BoxError("this box file was made for a different page image (SHA-256 does not match)")
                    state.boxes, state.imported, state.result = list(loaded.boxes), fingerprint, None
                    state.table_round += 1
                except UnicodeDecodeError:
                    st.error("The box file is not UTF-8 text.")
                except bx.BoxError as e:
                    st.error(f"The box file was not imported: {md(e)}")
            if state.imported == fingerprint:
                st.success(f"Imported {md(box_upload.name)}.")
    with tab_table:
        with st.form("box_table"):
            edited = st.data_editor(
                [{k: getattr(b, k) for k in bx.FIELDS} for b in state.boxes] or [{k: None for k in bx.FIELDS}],
                num_rows="dynamic", width="stretch", key=f"box_editor_{state.table_round}",
                column_config={"id": st.column_config.TextColumn("id", required=True),
                               **{k: st.column_config.NumberColumn(k, min_value=0, step=1, format="%d", required=True)
                                  for k in bx.INT_FIELDS}})
            if st.form_submit_button("Apply table"):
                try:
                    rows = [r for r in edited if any(v not in (None, "") for v in r.values())]
                    new = rows_to_boxes(rows)
                    bx.validate_boxes(new, page.width, page.height)
                    state.boxes, state.result = new, None
                    state.table_round += 1
                    st.rerun()
                except bx.BoxError as e:
                    st.error(f"The table was not applied: {md(e)}")
    boxes = list(state.boxes)
    if boxes:
        stem = Path(page.path).stem or "page"
        c1, c2, _ = st.columns([1, 1, 3])
        c1.download_button("Download boxes (JSON)", bx.dumps_json(boxes, page.width, page.height, page.sha256).encode("utf-8"),
                           file_name=f"{stem}_boxes.json", mime="application/json", width="stretch")
        c2.download_button("Download boxes (CSV)", bx.dumps_csv(boxes).encode("utf-8"),
                           file_name=f"{stem}_boxes.csv", mime="text/csv", width="stretch")

# ---- 3. preview
with st.container(border=True):
    st.subheader("3. Page preview")
    st.image(pio.draw_boxes(preview, boxes, scale) if boxes else preview, width="content")
    st.caption(f"{md(page.path)} · original {scale.original_width} × {scale.original_height} px · preview "
               f"{scale.preview_width} × {scale.preview_height} px · scale original/preview x {scale.scale_x:.4f}, "
               f"y {scale.scale_y:.4f} · {len(boxes)} box(es). The preview is for viewing only: boxes are stored and "
               "cropped in original pixels.")

# ---- 4. OCR
key = (page.sha256, tuple(boxes))
if state.result is not None and state.result["key"] != key:
    state.result = None                    # a result never stays on screen next to different boxes or another page

with st.container(border=True):
    st.subheader("4. OCR")
    run = st.button("Run OCR", type="primary", disabled=engine is None or not boxes)
    if run:
        try:
            with st.spinner(f"Recognizing {len(boxes)} box(es) ..."):
                devguard.assert_ocr_allowed(page.sha256)            # checked again at the moment the recognizer runs
                devguard.assert_ocr_allowed(devguard.development_page_for_pixels(page.gray))
                good, crops, skipped = [], [], []
                for b in boxes:
                    crop = bx.crop_from_page(page.gray, b)             # from the original page, no resampling
                    try:
                        validate_crop(crop)
                        good.append(b)
                        crops.append(crop)
                    except InvalidCropError as e:
                        skipped.append((b, str(e)))
                state.result = {"key": key, "boxes": good, "crops": crops, "skipped": skipped,
                                "recognitions": engine.recognize_batch(crops)}
        except devguard.DevelopmentBlocked:
            state.result = None
            st.error("This is a development page and the human-verified set is not frozen. No text was produced.")
        except Exception as e:
            state.result = None
            st.error(f"Recognition failed ({type(e).__name__}). No text was produced.")
    result = state.result
    if result is None:
        st.caption("Add boxes, then press Run OCR." if not boxes else "Press Run OCR to read the boxes.")
    else:
        for b, reason in result["skipped"]:
            st.warning(f"Box {md(b.id)} was not read: {md(reason)}")
        st.dataframe(
            [{"image": thumbnail(c), "box id": b.id, "text": r.text, "confidence": round(r.confidence, 3),
              "clipped": r.clipped, "line": b.line_index, "column": b.column_index,
              "x": b.x, "y": b.y, "w": b.w, "h": b.h, "width@48": r.width_px, "timesteps": r.timesteps}
             for b, c, r in zip(result["boxes"], result["crops"], result["recognitions"])],
            column_config={"image": st.column_config.ImageColumn("image", width="medium"),
                           "confidence": st.column_config.NumberColumn("confidence", format="%.3f")},
            width="stretch", hide_index=True, key="results_table")
        st.caption("Confidence is UNCALIBRATED: the mean top softmax probability over non-blank frames. Use it only to "
                   "decide which boxes to look at again; it is not a probability of being correct. "
                   "clipped = the box was wider than the model's limit at height 48 and was squeezed.")

if state.result is not None:
    result = state.result
    txt = export.to_txt(result["boxes"], result["recognitions"])
    doc = export.to_json(result["boxes"], result["recognitions"], page.sha256, engine.checkpoint_sha256,
                         engine.engine_version, page.path, page.width, page.height, result["skipped"])
    with st.container(border=True):
        st.subheader("5. Assembled text")
        st.text_area("Text in the operator's order (column, line, box order)", value=txt, height=260,
                     key=f"assembled_{hashlib.sha256(txt.encode('utf-8')).hexdigest()[:12]}")
        st.caption("Raw model predictions: no dictionary, language model or correction. Editing this field does not "
                   "change the downloads.")
        stem = Path(page.path).stem or "page"
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("Download TXT", txt.encode("utf-8"), file_name=f"{stem}_ocr.txt", mime="text/plain",
                           width="stretch", key="download_txt")
        d2.download_button("Download JSON", doc.encode("utf-8"), file_name=f"{stem}_ocr.json", mime="application/json",
                           width="stretch", key="download_json")

with st.expander("Technical information"):
    rows = [("Page file", md(page.path)), ("Page SHA-256", page.sha256), ("Page array", f"{page.shape} {page.dtype}, {page.format}"
             + (", EXIF rotation applied" if page.exif_transposed else ""))]
    if engine is not None:
        info = engine.info()
        rows += [("Model", info["model"]), ("Checkpoint", str(Path(info["checkpoint_path"]).relative_to(ROOT))),
                 ("Checkpoint SHA-256", info["checkpoint_sha256"] + " (verified: frozen Experiment 002-A)"),
                 ("Engine version", info["engine_version"]), ("Device", "GPU (CUDA)" if info["device"] == "cuda" else "CPU")]
    st.table({"item": [k for k, _ in rows], "value": [v for _, v in rows]})
