"""
Tamil OCR - local word and full-page recognizer (app.py, frontend V1.2). Inference only.

    .venv\\Scripts\\python -m streamlit run app.py          then open http://localhost:8501 ; stop with Ctrl+C

Word OCR: one cropped Tamil word per image in, the raw model prediction out; one image or a batch. An upload that the
layout check (wordocr.pagedetect) finds to be a page is refused there and sent to Full-page OCR. Full-page OCR: automatic
segmentation (pageocr.pipeline), word-by-word recognition, text rebuilt from the layout (wordocr.fullpage). TXT / JSON /
CSV export everywhere.
The page is presentation only. Validation, decoding, model loading, recognition and export live in the wordocr package,
which uses the project's canonical pieces unchanged: pageocr.imageio (decoder), pageocr.engine (input contract,
src/preprocess.py, CRNNv2, greedy CTC). No training, evaluation, language model, lexicon or post-correction. Nothing
is written to disk (results leave only through the download buttons) and no network service is contacted
(.streamlit/config.toml: localhost only, no usage statistics).
"""
import logging

import streamlit as st

from wordocr import SCOPE_STATEMENT, VERSION, checkpoints, export, fullpage, guard, inputs, page_export, pagedetect, service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("wordocr.app")

TITLE = "Tamil OCR — Word and Full-page Recognizer"
DEVICES = {"auto": "Auto (GPU if available)", "cpu": "CPU", "cuda": "GPU (CUDA)"}
EXT_LIST = ", ".join(sorted({e.upper() for e in inputs.EXTENSIONS}))


@st.cache_resource(max_entries=1, show_spinner=False)          # one model in memory; switching unloads the previous
def get_model(checkpoint_id: str, device_choice: str) -> service.LoadedModel:
    return service.load_model(checkpoint_id, device_choice)     # a failure is not cached: the next run tries again


@st.cache_data(max_entries=60, ttl=900, show_spinner=False)  # reruns do not decode the same upload again; a batch is at
                                                             # most 50 files; uploads leave the cache after 15 minutes
def decode(name: str, data: bytes):
    """DecodedImage, or a failed ItemResult with the user-facing reason. Never raises."""
    try:
        return inputs.decode_upload(name, data)
    except inputs.InputError as e:
        return service.failed(0, name, str(e))
    except Exception:
        log.exception("unexpected error while decoding %s", name)
        return service.failed(0, name, "The file could not be opened. Details are in the terminal running the app.")


@st.cache_resource(max_entries=4, ttl=900, show_spinner=False)   # the Segmentation of one page and one setting
def get_segmentation(sha256: str, k: float, gap: float, deskew: bool, _decoded, rule_removal: bool = False, clahe: bool = False):
    return fullpage.segment_page(_decoded, fullpage.PageSettings(sauvola_k=k, gap_floor_h=gap, deskew=deskew,
                                                                 exp_rule_removal=rule_removal, exp_contrast_clahe=clahe))


# Untrusted text (file names, predictions) placed in HTML inside st.markdown: every character that HTML or Markdown
# could interpret (tags, entities, links, images, emphasis) becomes a numeric entity, in one pass.
_ESC = str.maketrans({c: f"&#{ord(c)};" for c in "&<>\"'[]()!*_`~#|\\{}$"})


def esc(value) -> str:
    return str(value).translate(_ESC)


def tamil_box(text: str) -> None:
    st.markdown(f'<div class="ocr-text" lang="ta">{esc(text)}</div>', unsafe_allow_html=True)


def facts_table(rows) -> None:
    st.markdown('<table class="ocr-facts">' + "".join(
        f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in rows) + "</table>",
        unsafe_allow_html=True)


def results_table(results) -> None:
    """Batch results as an HTML table: selectable, wrapping Tamil text; every value escaped."""
    head = "<tr><th>#</th><th>File</th><th>Status</th><th>Recognized text</th><th>Note</th></tr>"
    rows = []
    for r in results:
        status = "failed" if not r.ok else ("empty" if r.empty else "ok")
        note = r.error or " ".join(r.image.warnings if r.image else ())
        rows.append(f'<tr class="st-{status}"><td>{r.index + 1}</td><td>{esc(r.name)}</td><td>{status}</td>'
                    f'<td class="ta" lang="ta">{esc(r.text)}</td><td>{esc(note)}</td></tr>')
    st.markdown(f'<table class="ocr-results">{head}{"".join(rows)}</table>', unsafe_allow_html=True)


def image_facts(d: inputs.DecodedImage) -> list:
    return [("File", d.name), ("Detected format", d.format), ("File size", f"{d.size_bytes:,} bytes"),
            ("Dimensions", f"{d.width} × {d.height} px"), ("Colour mode", d.mode),
            ("EXIF rotation applied", "yes" if d.exif_rotated else "no"),
            ("Transparency put on white", "yes" if d.transparency_flattened else "no"),
            ("Model input", f"{d.model_input_width} × 48 px" + (" (squeezed)" if d.clipped else "")),
            ("SHA-256", d.sha256[:16] + "…")]


def downloads(results, model_info: dict, key: str, csv: bool) -> None:
    cols = st.columns(3 if csv else 2)
    cols[0].download_button("Download TXT", export.to_txt(results), file_name=export.download_name(results, "txt"),
                            mime="text/plain", on_click="ignore", key=f"{key}_txt", width="stretch")
    cols[1].download_button("Download JSON", export.to_json(results, model_info), file_name=export.download_name(results, "json"),
                            mime="application/json", on_click="ignore", key=f"{key}_json", width="stretch")
    if csv:
        cols[2].download_button("Download CSV", export.to_csv(results), file_name=export.download_name(results, "csv"),
                                mime="text/csv", on_click="ignore", key=f"{key}_csv", width="stretch")


# ---------------------------------------------------------------- page setup

st.set_page_config(page_title=TITLE, page_icon="அ", layout="wide")
st.markdown("""
<style>
  html, body, [class*="css"], textarea, input { font-family: "Nirmala UI", "Latha", "Noto Sans Tamil", "Vijaya", "Segoe UI", sans-serif; }
  .block-container { padding-top: 2rem; max-width: 1150px; }
  .ocr-sub { color: #5f6368; font-size: 1.05rem; margin-top: -0.6rem; }
  .ocr-text { font-family: "Nirmala UI", "Latha", "Noto Sans Tamil", "Vijaya", sans-serif; font-size: 2.2rem; line-height: 1.6;
              padding: 0.6rem 1rem; border: 1px solid rgba(128,128,128,.35); border-radius: 0.5rem;
              overflow-wrap: anywhere; word-break: break-word; user-select: text; white-space: pre-wrap; }
  .ocr-facts { border-collapse: collapse; font-size: 0.92rem; }
  .ocr-facts td { padding: 0.12rem 1.1rem 0.12rem 0; vertical-align: top; overflow-wrap: anywhere; }
  .ocr-facts td:first-child { color: #6b7280; white-space: nowrap; }
  .ocr-results { width: 100%; border-collapse: collapse; table-layout: fixed; }
  .ocr-results th, .ocr-results td { text-align: left; padding: 0.35rem 0.6rem; border-bottom: 1px solid rgba(128,128,128,.3);
                                     vertical-align: top; overflow-wrap: anywhere; word-break: break-word; }
  .ocr-results th:nth-child(1) { width: 2.5rem; } .ocr-results th:nth-child(3) { width: 4.5rem; }
  .ocr-results th:nth-child(2) { width: 22%; } .ocr-results th:nth-child(5) { width: 30%; }
  .ocr-results td.ta { font-family: "Nirmala UI", "Latha", "Noto Sans Tamil", sans-serif; font-size: 1.35rem; user-select: text; }
  .ocr-results tr.st-failed td:nth-child(3) { color: #d93025; font-weight: 600; }
  .ocr-results tr.st-empty td:nth-child(3) { color: #b06000; font-weight: 600; }
  .ocr-page { font-family: "Nirmala UI", "Latha", "Noto Sans Tamil", "Vijaya", sans-serif; font-size: 1.3rem; line-height: 1.75;
              padding: 0.6rem 1rem; border: 1px solid rgba(128,128,128,.35); border-radius: 0.5rem; white-space: pre-wrap;
              overflow-wrap: anywhere; user-select: text; max-height: 32rem; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)

state = st.session_state
for k, v in (("single_round", 0), ("batch_round", 0), ("page_round", 0), ("single", None), ("batch", None), ("page", None)):
    state.setdefault(k, v)

# ---------------------------------------------------------------- sidebar: model and device

with st.sidebar:
    st.header("Model")
    present = {c.id: checkpoints.present(c) for c in checkpoints.REGISTRY}
    ckpt_id = st.selectbox("Checkpoint", checkpoints.ids(), index=checkpoints.ids().index(checkpoints.DEFAULT_ID),
                           format_func=lambda i: checkpoints.get(i).label + ("" if present[i] else "  (file missing)"),
                           key="checkpoint", help="Only checkpoints registered with the SHA-256 recorded at their "
                           "evaluation can be loaded. A missing or different file is refused, never replaced.")
    devices = ["auto", "cpu"] + (["cuda"] if service.cuda_available() else [])
    device_choice = st.radio("Device", devices, format_func=DEVICES.get, key="device", horizontal=False)
    if not service.cuda_available():
        st.caption("No CUDA GPU is available; the model runs on the CPU.")
    spec = checkpoints.get(ckpt_id)
    st.caption(f"Training data: {spec.training}.")

model = model_error = None
try:
    with st.spinner(f"Loading {spec.label} …"):
        model = get_model(ckpt_id, device_choice)
except service.ModelUnavailableError as e:
    model_error = str(e)
except Exception:
    log.exception("unexpected error while loading the model")
    model_error = "The model could not be loaded. Details are in the terminal running the app."
model_info = model.info() if model else None

# ---------------------------------------------------------------- header and status

st.title(TITLE)
st.markdown('<p class="ocr-sub">Word OCR reads one cropped Tamil word per image; Full-page OCR finds the words on a '
            'printed page and reads them one by one. The project\'s own trained CRNN model; it runs inside this app, with no '
            'external OCR service or API.</p>', unsafe_allow_html=True)
with st.container(border=True):
    c1, c2 = st.columns([3, 2])
    with c1:
        if model:
            st.success(f"Model ready: {model.spec.label} on {'GPU (CUDA)' if model.device == 'cuda' else 'CPU'}. "
                       "Checkpoint SHA-256 verified.")
        else:
            st.error(f"Model not available. {model_error}")
            if st.button("Retry loading", key="retry_model"):
                st.rerun()
    with c2:
        st.markdown(f"**Input:** {EXT_LIST} · up to {inputs.MAX_FILE_BYTES // (1024 * 1024)} MB · Word OCR: one word per "
                    f"image, up to {inputs.MAX_BATCH_FILES} images per batch · Full-page OCR: one printed page")
    st.caption(SCOPE_STATEMENT + " Full-page OCR therefore segments a page into word regions first and reads each one. "
               "The output is the raw model prediction: no dictionary, language model or spelling correction.")

tab_single, tab_batch, tab_page, tab_about = st.tabs(["Single image", "Batch", "Full-page OCR", "About and limits"])

# ---------------------------------------------------------------- single image

with tab_single:
    up = st.file_uploader(f"Image of one Tamil word ({EXT_LIST})", accept_multiple_files=False,
                          key=f"single_upload_{state.single_round}")
    item = decode(up.name, up.getvalue()) if up is not None else None
    layout = pagedetect.assess_cached(item.sha256, item.gray) if isinstance(item, inputs.DecodedImage) else None
    confirmed = False
    if layout is not None and layout.overridable:            # weak evidence only: the user may say it is one word
        confirmed = st.checkbox("This image shows ONE word: read it as a word anyway", value=False,
                                key=f"confirm_word_{item.sha256[:16]}",
                                help="Only for a single word that the layout check split into two rows (for example a "
                                     "blurred or broken word). Never for a page: use Full-page OCR.")
    is_page = layout is not None and layout.is_page and not confirmed    # a page is never read as one word
    run_key = ((item.sha256, ckpt_id, model.device if model else device_choice, confirmed)
               if isinstance(item, inputs.DecodedImage) else None)
    if state.single is not None and state.single["key"] != run_key:
        state.single = None                      # a result never stays next to a different image or model

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Image")
        if item is None:
            st.info("No image yet. Choose a file above.")
        elif isinstance(item, service.ItemResult):
            st.error(item.error)
        elif is_page:
            st.error(("This looks like more than one line of text. " if layout.overridable else "This looks like a page, not "
                      "one word. ") + pagedetect.refusal_message(layout))
            st.info("Open the **Full-page OCR** tab above and upload the same file there."
                    + (" If it is one word, tick the box above to read it anyway." if layout.overridable else ""))
            st.image(item.gray, caption="Uploaded image (not read as a word)", width="stretch")
            with st.expander("Image details"):
                facts_table(image_facts(item) + [("Layout check", f"{layout.lines} line(s), {layout.words} word region(s), "
                                                                   f"{layout.blocks} block(s)")])
        else:
            if layout is not None and layout.kind == "line":
                st.warning(f"{layout.reason} Word OCR reads the whole image as one sequence, without spaces between the "
                           "words. The Full-page OCR tab separates the words and reads them one by one.")
            st.image(item.gray, caption="What the model receives (grayscale)", width="stretch")
            st.image(inputs.model_view(item.gray), caption="At the model's input height (48 px)", width="stretch")
            for note in item.warnings:
                st.warning(note)
            with st.expander("Image details"):
                facts_table(image_facts(item))
    with right:
        st.subheader("Recognized text")
        done = state.single is not None
        b1, b2 = st.columns(2)
        run = b1.button("Run OCR", type="primary", width="stretch", key="run_single",
                        disabled=model is None or not isinstance(item, inputs.DecodedImage) or is_page or done,
                        help="Disabled when there is no valid image, the image is a page (use Full-page OCR), the model is "
                             "not loaded, or the result is shown already.")
        if b2.button("Clear", width="stretch", key="clear_single"):
            state.single = None
            state.single_round += 1
            st.rerun()
        if run and model is not None and isinstance(item, inputs.DecodedImage) and not is_page and state.single is None:
            with st.spinner("Recognizing …"):
                result = service.recognize_one(model, item, confirmed_single_word=confirmed)
                state.single = {"key": run_key, "result": result, "model": model_info}   # stored before any other UI call
            st.rerun()
        if state.single is None:
            st.caption("The recognized text appears here after Run OCR.")
        else:
            r, info = state.single["result"], state.single["model"]
            if not r.ok:
                st.error(f"No text was produced. {r.error}")
            elif r.empty:
                st.warning("The model produced no characters for this image. Check that it shows one clear Tamil word "
                           "with a little margin around it.")
            else:
                tamil_box(r.text)
                st.code(r.text, language=None, wrap_lines=True)
                st.caption(f"{len(r.text)} Unicode code points · read by {info['checkpoint_label']} on "
                           f"{info['device'].upper()} · use the copy icon above, or select the text.")
            if r.ok:
                downloads([r], info, "single", csv=False)
                with st.expander("Technical details"):
                    facts_table([("Checkpoint", f"{info['checkpoint_label']} ({info['checkpoint_path']})"),
                                 ("Checkpoint SHA-256", info["checkpoint_sha256"]), ("Epoch", info["checkpoint_epoch"]),
                                 ("Device", info["device"]), ("Output frames", r.timesteps),
                                 ("Recognition time", f"{r.seconds:.3f} s"),
                                 ("Uncalibrated score", f"{r.uncalibrated_score:.3f}")])
                    st.caption(export.SCORE_NOTE)

# ---------------------------------------------------------------- batch

with tab_batch:
    ups = st.file_uploader(f"Images, one Tamil word each ({EXT_LIST}; at most {inputs.MAX_BATCH_FILES})",
                           accept_multiple_files=True, key=f"batch_upload_{state.batch_round}") or []
    too_many = len(ups) > inputs.MAX_BATCH_FILES
    items = [] if too_many else [decode(u.name, u.getvalue()) for u in ups]
    batch_key = (tuple((u.name, getattr(i, "sha256", None)) for u, i in zip(ups, items)), ckpt_id,
                 model.device if model else device_choice) if items else None
    if state.batch is not None and state.batch["key"] != batch_key:
        state.batch = None
    if too_many:
        st.error(f"{len(ups)} files were chosen; the limit is {inputs.MAX_BATCH_FILES} per batch. Remove some files.")
    elif items:
        n_ok = sum(isinstance(i, inputs.DecodedImage) for i in items)
        st.caption(f"{len(items)} file(s): {n_ok} ready, {len(items) - n_ok} cannot be read (listed in the results).")
    b1, b2 = st.columns([1, 1])
    run_b = b1.button("Run OCR on all", type="primary", width="stretch", key="run_batch",
                      disabled=model is None or not any(isinstance(i, inputs.DecodedImage) for i in items) or state.batch is not None)
    if b2.button("Clear batch", width="stretch", key="clear_batch"):
        state.batch = None
        state.batch_round += 1
        st.rerun()
    if run_b and model is not None and state.batch is None:
        with st.spinner(f"Recognizing {len(items)} image(s) …"):
            results = service.recognize(model, items)
            state.batch = {"key": batch_key, "results": results, "model": model_info}
        st.rerun()
    if state.batch is not None:
        results, info = state.batch["results"], state.batch["model"]
        n_err = sum(not r.ok for r in results)
        n_empty = sum(r.empty for r in results)
        st.markdown(f"**{len(results) - n_err} of {len(results)} recognized** · {n_err} failed · {n_empty} empty · "
                    f"{info['checkpoint_label']} on {info['device'].upper()}")
        results_table(results)
        downloads(results, info, "batch", csv=True)
    elif not items:
        st.info("No images yet. Choose one or more files above.")

# ---------------------------------------------------------------- full page

with tab_page:
    st.caption(page_export.PAGE_STATEMENT)
    pup = st.file_uploader(f"Image of one printed Tamil page ({EXT_LIST})", accept_multiple_files=False,
                           key=f"page_upload_{state.page_round}")
    with st.expander("Segmentation settings"):
        st.caption("Defaults suit most clean printed pages. A change re-runs the segmentation; press Read page again.")
        sc1, sc2 = st.columns(2)
        auto_margin = sc1.checkbox("Automatic crop margin (25 % of the text height)", value=True, key="page_margin_auto")
        margin_px = sc1.number_input("Crop margin (pixels)", min_value=0, max_value=64, value=6, step=1, key="page_margin_px",
                                     disabled=auto_margin, help="White space added around each word box when its crop is cut.")
        deskew = sc1.checkbox("Correct page skew (up to 5°)", value=True, key="page_deskew")
        k_val = sc2.slider("Ink threshold sensitivity (Sauvola k)", 0.10, 0.45, float(fullpage.DEFAULTS.sauvola_k), 0.01,
                           key="page_k", help="Lower finds fainter ink; higher keeps only darker ink.")
        gap_val = sc2.slider("Smallest word gap (× line height)", 0.08, 0.40, float(fullpage.DEFAULTS.gap_floor_h), 0.01,
                             key="page_gap", help="Higher joins more neighbouring pieces into one word; lower splits more.")
        st.caption("Experimental options (off by default; see reports/diagnostic_implementation for the evidence behind each)")
        ex1, ex2, ex3 = st.columns(3)
        exp_rule = ex1.checkbox("Remove ruling lines (segmentation)", value=False, key="page_exp_rule",
                                help="Drops long horizontal rule pixels before words are found. Notebook pages.")
        exp_clahe = ex2.checkbox("Local contrast equalisation", value=False, key="page_exp_clahe")
        exp_fill = ex3.checkbox("Fill ruling in crops (changes model input)", value=False, key="page_exp_fill")
    diag_mode = st.checkbox("Diagnostic mode (region table, crops, warnings, evidence export)", value=False, key="page_diag")
    settings = fullpage.PageSettings(margin="auto" if auto_margin else int(margin_px), sauvola_k=round(float(k_val), 3),
                                     gap_floor_h=round(float(gap_val), 3), deskew=bool(deskew),
                                     exp_rule_removal=bool(exp_rule), exp_contrast_clahe=bool(exp_clahe), exp_crop_rule_fill=bool(exp_fill))
    pitem = decode(pup.name, pup.getvalue()) if pup is not None else None
    seg = seg_error = None
    if isinstance(pitem, inputs.DecodedImage):
        try:
            with st.spinner("Finding text blocks, lines and words …"):
                seg = get_segmentation(pitem.sha256, settings.sauvola_k, settings.gap_floor_h, settings.deskew, pitem,
                                       settings.exp_rule_removal, settings.exp_contrast_clahe)
        except fullpage.PageError as e:
            seg_error = str(e)
    page_key = ((pitem.sha256, tuple(sorted(settings.as_dict().items())), ckpt_id, model.device if model else device_choice)
                if seg is not None else None)
    if state.page is not None and state.page["key"] != page_key:
        state.page = None                          # results never stay next to another page, setting or model
    pr = state.page["result"] if state.page is not None else None

    p1, p2 = st.columns([1, 1])
    read = p1.button(f"Read page ({len(seg.words)} word regions)" if seg is not None else "Read page", type="primary",
                     width="stretch", key="run_page",
                     disabled=model is None or seg is None or len(seg.words) == 0 or pr is not None,
                     help="Disabled when there is no readable page, no word region was found, the model is not loaded, "
                          "or the result is shown already.")
    if p2.button("Clear page", width="stretch", key="clear_page"):
        state.page = None
        state.page_round += 1
        st.rerun()

    if pitem is None:
        st.info("No page yet. Choose an image of a printed Tamil page above.")
    elif isinstance(pitem, service.ItemResult):
        st.error(pitem.error)
    elif seg_error:
        st.error(seg_error)
    else:
        summary = fullpage.segmentation_summary(seg)
        st.markdown(f"**Layout found:** {summary['blocks']} block(s) · {summary['lines']} line(s) · {summary['word_regions']} "
                    f"word region(s) · text height {summary['text_height_px']:.0f} px · skew {abs(summary['skew_deg']) if abs(summary['skew_deg']) < 0.005 else summary['skew_deg']:.2f}°"
                    + (" (corrected)" if summary["deskew_applied"] else ""))
        for flag in summary["flags"]:
            st.warning(f"Segmentation note: {flag}")
        if not seg.words:
            st.error("No word regions were found on this page, so nothing can be read. Check that it shows dark printed "
                     "text on a light background, or lower the ink threshold sensitivity in the settings.")
        if read and model is not None and seg.words and pr is None:
            bar = st.progress(0.0, text="Reading word regions …")
            try:
                result = fullpage.recognize_page(model, pitem, seg, settings,
                                                 progress=lambda d, n: bar.progress(d / n, text=f"Reading word regions … {d} of {n}"))
                state.page = {"key": page_key, "result": result}
                st.rerun()
            except guard.ProtectedDataError as e:
                st.error(str(e))
            except Exception:
                log.exception("full-page recognition failed")
                st.error("The page could not be read. Details are in the terminal running the app.")
        pr = state.page["result"] if state.page is not None else None
        v1, v2 = st.columns(2, gap="large")
        with v1:
            st.subheader("Page")
            st.image(pitem.gray, caption=f"Uploaded page, {pitem.width} × {pitem.height} px (grayscale, as the crops are cut)",
                     width="stretch")
            with st.expander("Image details"):
                facts_table(image_facts(pitem))
        with v2:
            st.subheader("Segmentation")
            st.image(fullpage.overlay(seg, pr.words if pr else None), width="stretch",
                     caption="Blue: text blocks in reading order · green: lines · red: word regions, numbered in reading "
                             "order (orange: not read)" + (" · shown deskewed" if summary["deskew_applied"] else ""))
        if pr is None:
            st.caption("Check the segmentation, then press Read page. Each word region is cut from the page and read by the "
                       "selected model on its own.")
        else:
            c = pr.counts
            st.markdown(f"**{c['ok']} of {c['regions']} word regions read** · {c['empty']} empty · {c['failed']} failed · "
                        f"{pr.model['checkpoint_label']} on {pr.model['device'].upper()} · crop margin {pr.margin_px} px · "
                        f"{pr.seconds['recognition']:.1f} s")
            st.subheader("Page text")
            if pr.text:
                page_html = "<br>".join(esc(line) for line in pr.text.split(chr(10)))     # explicit breaks: Markdown would re-wrap blank lines
                st.markdown(f'<div class="ocr-page" lang="ta">{page_html}</div>', unsafe_allow_html=True)
                st.code(pr.text, language=None, wrap_lines=True)
            else:
                st.warning("No text was produced for this page.")
            st.caption("Rebuilt from the layout: words of a line joined by a space, lines by a line break, blocks by a blank "
                       "line. Every word is the raw prediction of one numbered region below; nothing is corrected.")
            d1, d2, d3 = st.columns(3)
            d1.download_button("Download TXT", page_export.to_txt(pr), file_name=page_export.download_name(pr, "txt"),
                               mime="text/plain", on_click="ignore", key="page_txt", width="stretch")
            d2.download_button("Download JSON", page_export.to_json(pr), file_name=page_export.download_name(pr, "json"),
                               mime="application/json", on_click="ignore", key="page_json", width="stretch")
            d3.download_button("Download CSV", page_export.to_csv(pr), file_name=page_export.download_name(pr, "csv"),
                               mime="text/csv", on_click="ignore", key="page_csv", width="stretch")
            failed_words = [w for w in pr.words if not w.ok]
            if failed_words:
                st.warning(f"{len(failed_words)} word region(s) could not be read: "
                           + "; ".join(f"#{w.index + 1} ({w.error})" for w in failed_words[:12])
                           + (" …" if len(failed_words) > 12 else ""))
            with st.expander(f"Word regions ({len(pr.words)})"):
                head = ("<tr><th>#</th><th>Region</th><th>Line</th><th>Status</th><th>Recognized text</th>"
                        "<th>Box (x1, y1, x2, y2)</th></tr>")
                rows = "".join(f'<tr class="st-{w.status}"><td>{w.index + 1}</td><td>{esc(w.id)}</td>'
                               f"<td>{w.block + 1}.{w.line + 1}</td><td>{w.status}</td>"
                               f'<td class="ta" lang="ta">{esc(w.text)}</td><td>{esc(", ".join(map(str, w.box)))}</td></tr>'
                               for w in pr.words)
                st.markdown(f'<table class="ocr-results ocr-words">{head}{rows}</table>', unsafe_allow_html=True)
            with st.expander("Word crops as the model received them (first 60)"):
                crops = fullpage.crop_images(seg, settings)
                cols = st.columns(4)
                for i, (w, crop) in enumerate(zip(pr.words[:60], crops[:60])):
                    cols[i % 4].image(crop, caption=f"#{w.index + 1}: {w.text if w.ok else '(' + w.status + ')'}")
            with st.expander("Processing details"):
                facts_table([("Model", f"{pr.model['checkpoint_label']} ({pr.model['checkpoint_path']})"),
                             ("Checkpoint SHA-256", pr.model["checkpoint_sha256"]), ("Device", pr.model["device"]),
                             ("Segmentation", pr.segmentation["pipeline_version"]),
                             ("Settings", ", ".join(f"{k} = {v}" for k, v in pr.settings.items())),
                             ("Crop margin used", f"{pr.margin_px} px"),
                             ("Steps", " → ".join(pr.segmentation["steps"])),
                             ("Removed components", ", ".join(f"{k}: {v}" for k, v in pr.segmentation["removed_components"].items()) or "none")])
                st.caption(export.SCORE_NOTE)
            if diag_mode:
                from wordocr import diagnostics_view
                try:
                    diagnostics_view.render(model, pitem, pup.getvalue(), seg, pr, settings)
                except Exception:
                    log.exception("diagnostic view failed")
                    st.error("The diagnostic view failed; the OCR result above is unaffected. Details are in the terminal.")

# ---------------------------------------------------------------- about

with tab_about:
    st.markdown(f"""
**Word OCR** (Single image, Batch) reads one cropped image of a single Tamil word and returns the model's raw
prediction (greedy CTC decoding). The model knows 100 characters: the Tamil letters, vowel signs and digits, ASCII
digits and some punctuation. It has **no space character**, so it cannot read a line or a page in one pass. Before a
word is read, a layout check looks for several lines or text blocks; an image that is a page is **not** read as a word
(it would come out as one or two characters) and is sent to Full-page OCR instead.

**Full-page OCR** finds text blocks, columns, lines and word regions on a printed page with classical image processing
(illumination correction, local thresholding, cleaning, deskew up to 5°, column gutters, line bands, word gaps), cuts
each word region from the page with a small margin, reads every region separately with the selected model, and rebuilds
the text from the layout: spaces between words, line breaks between lines, a blank line between blocks. Every word can
be traced to its numbered region and crop. **The segmentation has been tested on generated pages only; its accuracy on
real printed pages has not been measured.** It does not handle curved or photographed pages with strong perspective,
handwriting, tables, mixed scripts in one word, or thresholded black-and-white scans.

**What it does not do.** No PDF input; no dictionary, language model or spelling correction; no training or
evaluation. The research tools `app_page.py` (operator-supplied word boxes) and `app_page_auto.py` (segmentation
preview of the research pipeline) are separate.

**How to prepare an image.**
- Word OCR: one word per image, cropped with a small margin of background around it. Full-page OCR: one page per image.
- Horizontal text, dark ink on a light background, as photographed or scanned.
- Not thresholded to pure black and white: images with only two gray levels are refused.
- The model reads the image at a height of 48 px; text taller than about 20 px is plenty.
- {EXT_LIST}; at most {inputs.MAX_FILE_BYTES // (1024 * 1024)} MB and {inputs.MAX_PIXELS:,} pixels. The file extension
  must match the content.

**Results can be wrong**, especially for images unlike the training data (IIIT-HW handwritten words and synthetic
printed words). Recorded test results of the checkpoints describe their test sets, not your images.

**The score** shown under Technical details and in the exports is not a confidence or an accuracy:
{export.SCORE_NOTE}

**Privacy.** Uploaded images and results are held in the memory of the computer or server that runs this app, for the
browser session (decoded uploads stay in its cache for at most 15 minutes). They are not written to its disk and are
not sent to any other service. When the app is hosted publicly, your image is uploaded to that server: do not upload
private or confidential documents. Downloads are saved by your browser.

**Version.** {VERSION}; checkpoints are opened read-only and verified by SHA-256 before loading.
""")
    st.table([{"Checkpoint": c.label, "File": c.path, "Training data": c.training,
               "Available": "yes" if present[c.id] else "file missing", "SHA-256": c.sha256[:16] + "…"}
              for c in checkpoints.REGISTRY])
