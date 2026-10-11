# Architecture

Tamil OCR V2 has two independent parts:

- a **word recognizer** (a neural network) that reads one cropped word at a time;
- a **page segmenter** (classical image processing, no learning) that finds the words on a page.

Full-page OCR is segmentation followed by word-by-word recognition. All module paths below are real.

```
                         ┌─────────────────────────── app.py (Streamlit) ───────────────────────────┐
 upload ─► wordocr/inputs.py ─► wordocr/guard.py ─► wordocr/pagedetect.py
   (decode, gray, checks)        (protected files)    (word, line or page?)
                                                          │
            ┌──────────── word ───────────────────────────┴──────────────── page ─────────────┐
            ▼                                                                                 ▼
   wordocr/service.py                                                       wordocr/fullpage.py
   load_model() ─► wordocr/checkpoints.py (hash-pinned registry)            segment_page() ─► pageocr/pipeline.py
   recognize_one() ─► pageocr/engine.py (CRNNv2 + greedy CTC)               recognize_page() ─► pageocr/engine.py, one crop per word
            │                                                                                 │
            ▼                                                                                 ▼
   wordocr/export.py (TXT/JSON/CSV)                    wordocr/page_export.py (TXT/JSON/CSV), wordocr/diagnostics*.py
```

## 1. Input and safety (`wordocr/`)

| Module | Role |
|---|---|
| `inputs.py` | decodes one upload to a 2-D uint8 grayscale image (EXIF orientation applied) and records its SHA-256; refuses empty, unreadable, binarised and PDF input, files over 20 MB, images over 40 megapixels or 12,000 px a side, and batches over 50 files |
| `guard.py`, `pageocr/protected.py`, `pageocr/protected_manifest.json` | refuses the project's protected research files by size and SHA-256 fingerprint; fails closed if the manifest is missing or altered; `TAMIL_OCR_PROTECTED_ROOTS` adds local folders |
| `pageocr/devguard.py`, `pageocr/rights.py` | development-page guard and redistribution-status rules (research workflow) |
| `pagedetect.py` | decides from the image's own layout whether an upload is one word, a line or a page; a page sent to word OCR is refused |

## 2. Word recognition

| Module | Role |
|---|---|
| `src/charset.py` | the single character set: 100 characters (Tamil letters and signs, digits, punctuation) plus the CTC blank (index 0), giving 101 classes |
| `src/preprocess.py` | the **one** preprocessing path for training, evaluation and the app: grayscale, height 48 px (aspect kept), width 16–1024 px (wider crops are squeezed), normalised to [−1, 1] |
| `src/model/crnn_v2.py` | **CRNNv2**: residual CNN (channels 64-128-256-384), 2-layer bidirectional LSTM (hidden 256), linear layer to 101 classes; 11,038,277 parameters; trained from scratch |
| `pageocr/engine.py` | `OcrEngine`: loads a checkpoint only if its SHA-256 matches, with `torch.load(weights_only=True)`, and checks architecture and character set; batched greedy CTC decoding; reports an *uncalibrated* score (mean winning-class probability) |
| `wordocr/checkpoints.py` | registry of the four loadable checkpoints (ID, path, SHA-256, provenance); default `002a`; never substitutes another file |
| `wordocr/service.py` | `load_model`, `recognize_one`, `recognize` (a list): no Streamlit; usable from scripts |

Decoding is greedy (best path), with no dictionary or language model. The model has no space character; spaces in page text come from the layout.

## 3. Page segmentation (`pageocr/`, no model)

Two image paths, kept apart on purpose (`pageocr/layout.py`):

- the **OCR image**: the original grayscale, optionally deskewed. Every pixel the recognizer receives comes from here.
- the **layout image**: a boolean ink mask, used only to *find* text.

| Step | Module | What happens |
|---|---|---|
| 1. Layout image | `layout.py` | illumination correction; Sauvola thresholding; removal of specks, page borders, rule lines, bleed-through and figures (each counted) |
| 2. Skew | `deskew.py` | angle that sharpens the row profile, within ±5°; if at least 0.2°, the OCR image is rotated once and the layout image rebuilt from it |
| 3. Blocks | `columns.py` | columns by whitespace gutters (horizontal-then-vertical XY cut) |
| 4. Lines | `lines.py` | horizontal bands of ink rows; thin mark bands joined to their line; over-tall bands cut at the weakest rows (flagged as guesses) |
| 5. Words | `words.py` | components grouped by overlap; marks (pulli, dots) attached; word gaps from each line's own gap distribution, or the page's |
| 6. Reading order | `reading_order.py` | IDs `bNN_lNNN_wNN`; boxes in both frames (`OcrImage.to_original` is the exact inverse affine) |
| 7. Crops | `pipeline.ocr_crops` | tight word box plus margin, cut from the OCR image only (a layout image is refused) |

`pipeline.segment()` runs steps 1–6 and returns a `Segmentation`; `pipeline.diagnostics()` returns it as plain data. Parameters are in `SegParams`, in units of the page's text height.

**Experimental options:** all **off by default**, all opt-in. With all of them off, the output is identical to the default pipeline (verified on 32 inputs and 480 regions).

| Option | Module | Status |
|---|---|---|
| `exp_rule_removal` | `rules.py` | candidate on generated pages |
| `exp_contrast_clahe` | `layout.py` | rejected |
| `exp_crop_rule_fill` | `wordocr/fullpage.py` | candidate; changes model input |
| `exp_angle_lines` | `slant_lines.py` | rejected |

## 4. Full-page OCR and outputs (`wordocr/`)

| Module | Role |
|---|---|
| `fullpage.py` | `PageSettings` (user settings to `SegParams`; margin); `segment_page` (inverts a light-on-dark page first, exactly and reversibly); `recognize_page`, which reads every crop and rebuilds the text in reading order; limits: text height ≥ 5 px, at most 3,000 regions |
| `page_export.py`, `export.py` | TXT, JSON and CSV exports built from the same result objects the screen shows |
| `diagnostics.py` | read-only analysis of a run: region records, 11 heuristic warnings, overlays, evidence bundle (ZIP or folder; crops re-cut and hash-checked) |
| `diagnostics_view.py` | the diagnostic panel |
| `pageocr/line_geometry.py` | display-only, angle-aware line boxes: a robust fit of each line's direction, with curvature and uncertainty flags |
| `crop_compare.py` | condition A (manual box) vs B (pipeline crop), the same recognizer call; crop-integrity verdicts; only confirmed references are scored |

## 5. Synthetic data and training (`src/`)

| Module | Role |
|---|---|
| `src/synth/fonts.py` | discovers fonts in `assets/fonts/` and checks their Tamil glyph coverage |
| `src/synth/render.py` | HarfBuzz shaping and FreeType rasterisation (correct Tamil conjuncts and vowel signs) |
| `src/synth/text_sampler.py` | draws words from the word list (`data/text_corpus/…`, not in the repository) |
| `src/synth/degrade.py`, `generator.py` | modern- and old-print degradations; `SynthWordGenerator` |
| `src/augment.py` | augmentation of real word images |
| `src/data/datasets.py` | `ManifestDataset` (jsonl manifests), `SynthSource`, `MixtureDataset`, width-bucket and paired-source batch samplers, `collate` (CTC targets) |
| `src/train.py` | `python -m src.train --config configs/<x>.yaml`: seeding, `build_model`, AdamW with a one-cycle schedule, mixed precision, `TrainStep` (CTC), validation CER, `best.pth` / `last.pth`, history and log |
| `src/metrics.py` | micro-averaged CER, WER, word accuracy, per source |

Data preparation, evaluation and audits are in `tools/`. Segmentation experiments are in `dev_tools/`: `synthetic_conditions.py`, `segmentation_experiments.py`, `slant_experiments.py`, `compare_manual_crops.py`, `experiment_record.py`.

## 6. Repository layout

```
app.py, app_page.py, app_page_auto.py   Streamlit apps (app.py is the supported one)
wordocr/   pageocr/   src/               application, segmentation, model and training code
configs/   runs/<experiment>/            training configurations; per-run config, history, log (weights not in Git)
assets/fonts/                            fonts with their OFL licence texts
data/README.md, data/manifests/          dataset policy and SHA-256 manifest (data itself not in Git)
tools/   dev_tools/                      data, evaluation, verification and experiment scripts
tests/                                   unit, app and browser tests (generated inputs)
reports/                                 experiment records and the Experiment 002-A evaluation report
docs/                                    guides (installation, training, evaluation, dependencies, security, this file)
.github/workflows/                       CI: unit tests without a checkpoint, repository check
```

## 7. Tests

`run_all_tests.py` runs the suites (`--list` shows them):

- **Unit suites:** `guard`, `pageocr`, `word`, `fullpage`, `diagnostics`, `training_code`.
- **Browser suites:** `word_browser`, `fullpage_browser` and `diag_browser` drive the real server in headless Microsoft Edge, and need Node.js.

Without the checkpoint, the model-dependent modules are reported as NOT RUN. Results of the repository preparation: `REPOSITORY_AUDIT.md`.
