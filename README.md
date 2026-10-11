# Tamil OCR V2

A local Tamil text recognizer with two parts:

- **Word recognition:** a custom word-recognition network (CRNNv2 with CTC decoding), trained from scratch by this project.
- **Full-page segmentation:** classical image processing (no neural network) that finds the words on a page.

A Streamlit interface runs both, plus a diagnostic view for analysing segmentation errors. Everything runs on your own computer: no account, API key, cloud service, external OCR engine or pretrained model is used.

> **Trained weights are not included in this repository.** The application needs a compatible, authorised local checkpoint to recognise text; see [Model checkpoints](#model-checkpoints). Without one, it starts and says clearly that no model is available. It never downloads a model and never substitutes another one.

## What it does

| Function | What happens | Code |
|---|---|---|
| **Word OCR** | reads one image of a single Tamil word, or a batch of such images | `app.py` (tabs *Single image*, *Batch*), `wordocr/service.py` |
| **Full-page OCR** | finds columns, lines and words on a page image; reads each word; rebuilds the text in reading order | `app.py` (tab *Full-page OCR*), `wordocr/fullpage.py`, `pageocr/` |
| **Diagnostic mode** (opt-in) | overlays of lines and word regions, a sortable region table, single-region crop inspection, heuristic warnings, a reproducible evidence bundle (ZIP), manual-versus-pipeline crop comparison, and angle-aware line boxes | `wordocr/diagnostics.py`, `wordocr/diagnostics_view.py`, `wordocr/crop_compare.py`, `pageocr/line_geometry.py` |

Recognition and segmentation are separate. The neural network only ever sees one word crop at a time. Segmentation (blocks, lines, words, reading order) uses thresholding, connected components and geometry. A wrong page result can come from either part; the diagnostic mode exists to tell which.

## Architecture

```
page image ─► pageocr (segmentation, no model)  ─► word boxes ─► word crops (original grayscale)
                 illumination correction, Sauvola thresholding,             │
                 component cleaning, deskew, columns, lines,                ▼
                 word gaps, reading order                         CRNNv2 + greedy CTC (src/model, pageocr/engine.py)
                                                                            │
                                                         text per word ─► page text rebuilt from the layout
```

- **Model** (`src/model/crnn_v2.py`): a convolutional feature extractor, a 2-layer bidirectional LSTM and a CTC output over a 101-class Tamil character set (`src/charset.py`), about 11 million parameters. Input: one word crop, height 48 px (`src/preprocess.py`).
- **Checkpoint registry** (`wordocr/checkpoints.py`): every loadable checkpoint is pinned by its SHA-256. A file with another hash is refused; no other file is ever substituted.
- **Protected-data guard** (`wordocr/guard.py`, `pageocr/protected.py`): refuses inputs that are the project's protected research files, recognised by fingerprints in `pageocr/protected_manifest.json` (sizes and hashes only). It fails closed: if the manifest is missing or altered, every upload is refused.
- **Training** (`src/train.py`, `configs/`): mixed real-handwriting and synthetic-print training with a HarfBuzz-based Tamil text renderer (`src/synth/`).

## Installation

Tested with Python 3.12 (64-bit) on Windows 11. Linux and macOS were not tested; the code uses no Windows-only feature.

```
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # or .../whl/cpu for CPU only
pip install -r requirements-runtime.txt     # the application
pip install -r requirements-test.txt        # also: tests, training tools, synthetic generator
```

`docs/DEPENDENCIES.md` explains every package. Internet access is needed only to install the packages.

## Model checkpoints

No checkpoint is distributed. The registry expects the default model at:

```
runs/experiment_002a/best.pth      SHA-256 fb29131ed71fe0c89229497d0171b1ff2de4e1e581b4d8c1e09d6fbbaf62d5f1
```

- Place an authorised copy of that file there. `*.pth` files are ignored by `.gitignore`, so they are never committed.
- Only files whose SHA-256 matches the registry are loaded. A checkpoint you train yourself has a different hash: it needs its own registry entry in `wordocr/checkpoints.py` (see `docs/MODEL_CHECKPOINTS.md`).
- Without a checkpoint, the application shows "Model not available. The checkpoint file … was not found … (it is not downloaded automatically)". Segmentation still runs, but nothing is recognised.

## Run the application

```
python -m streamlit run app.py
```

Open `http://localhost:8501`. The server listens on this computer only, and usage statistics are off (`.streamlit/config.toml`).

For the diagnostic view: *Full-page OCR* tab → tick **Diagnostic mode** → *Read page*. The "Line boxes" control switches between angle-aware line boxes and the horizontal bands the segmentation actually used.

`app_page.py` (you supply the word boxes) and `app_page_auto.py` (automatic segmentation with more settings) are earlier interfaces; `app.py` is the supported one.

## Run the tests

```
python -B run_all_tests.py --no-browser      # unit suites
python -B run_all_tests.py                   # plus 3 browser suites (need Node.js and Microsoft Edge)
python -B run_all_tests.py --list            # show the suites
```

All test inputs are generated in memory; no dataset is needed and nothing is downloaded.

**Without a checkpoint**, the runner runs the suites that do not need one and reports the model-dependent suites as **NOT RUN** (exit code 2), naming the missing file. It does not report them as passed. With the registered checkpoint in place, every suite runs. The results recorded for this export are in `REPOSITORY_AUDIT.md`.

## Training (optional)

`src/train.py` with a configuration from `configs/` reproduces the documented experiments, given the training data. **The data is not included:**

- the IIIT-INDIC-HW-WORDS Tamil handwriting dataset, a third-party dataset you must obtain under its own terms (`tools/import_hf_parquet.py` converts a local parquet copy; nothing is downloaded);
- the word list the synthetic generator samples from (`data/text_corpus/…`).

`docs/TRAINING_GUIDE.md` lists exactly what is needed. A model you train is yours to register; do not overwrite the registered files.

## Segmentation experiments: what they show and what they do not

Controlled experiments on **generated** pages (printed Tamil fonts, known word boxes) are summarised in `docs/EXPERIMENTS.md`, with the scripts in `dev_tools/`:

- **Recognition vs segmentation.** Words cut with their exact boxes were read correctly in every tested condition. The pipeline's own crops fail mainly on notebook-ruled pages. On these pages the failures are segmentation, not recognition.
- **Rule removal** (`exp_rule_removal`, opt-in): removes horizontal ruling before segmentation; on generated ruled pages, page CER fell from 27–91 % to 0–1.6 % on a fresh held-out split. It is **off by default** until it has been tested on real pages.
- **Rejected:** contrast equalisation (CLAHE), a wider crop margin, and angle-aware line tracking (`exp_angle_lines`; large gains on pages whose lines have different angles, but regressions on curved and ruled pages). All three stay off.

They do **not** show accuracy on real documents or real handwriting. The fonts are training fonts, the ruling is simulated, and no real page has been evaluated.

## Status

- **Real-document accuracy has not been established.** The available numbers are a handwriting dataset's test split and synthetic print (`docs/EVALUATION_AND_LIMITATIONS.md`). The next planned step is a manual-versus-pipeline comparison on a real page with human-verified transcriptions (`dev_tools/compare_manual_crops.py`).
- **Retraining is not planned:** the synthetic evidence points at segmentation, and no authorised labelled real-document set with a held-out split exists yet.
- The output is the raw model prediction, without dictionary correction. The displayed score is not a probability of being correct. PDF input is not supported.

## Not included

| Not included | Why |
|---|---|
| Model weights (`*.pth` and any other weight format) | not distributed; supply an authorised local copy |
| Datasets, word lists, real or protected research pages and their annotations | third-party terms or protected material |
| Experiment outputs, logs, evidence bundles, diagnostic runs | generated; reproducible with the scripts |
| Release and workspace tooling of the private project history | not needed to use or develop the code |
| Credentials, environments, caches | never part of the project |

The complete per-file record is `EXPORT_MANIFEST.csv`; checks and open issues are in `REPOSITORY_AUDIT.md`.

## Licence

**No licence has been chosen for this project's code and documentation yet**; see `LICENSE_SELECTION_REQUIRED.md`. Until one is added, no reuse rights are granted.

The fonts in `assets/fonts/` are third-party, under the SIL Open Font License 1.1; each family's licence text is next to it. Trained weights depend on the terms of the third-party training dataset.

## Further documents

- `docs/INSTALLATION.md`, `docs/TRAINING_GUIDE.md`, `docs/MODEL_PROVENANCE.md`, `docs/EVALUATION_AND_LIMITATIONS.md`, `docs/DEPENDENCIES.md`, `docs/SECURITY_AND_PRIVACY.md`: written for the project's 2026-10-10 review release, each with a note on what differs here.
- `docs/MODEL_CHECKPOINTS.md`, `docs/EXPERIMENTS.md`.
- `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`.
