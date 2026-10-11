> **Note for this repository.** This document was written for the project's private 2026-10-10 review release and is kept unchanged below this note. Differences here: trained weights (`runs/*/best.pth`) are **not included** (see `docs/MODEL_CHECKPOINTS.md`); the release evidence files (`RELEASE_*`, `RELEASE_SHA256SUMS.txt`), `release_tools/`, `release_evidence/`, `GLOBAL_CLEANUP_PLAN.md` and `CHANGELOG_CLEAN_RELEASE.md` are not part of this repository; the other release documents now live in `docs/`. The repository's own record is `EXPORT_MANIFEST.csv` and `REPOSITORY_AUDIT.md`.

# Dependencies

The requirement files of this release were written from the imports of the retained source files and confirmed by the test runs, not copied from a package listing of the development environment. The full table, one row per package, is `RELEASE_DEPENDENCY_INVENTORY.csv`.

## Required

Versions are those of the environment the release was tested in (Python 3.12.10, Windows 11). They are pinned because they are the verified ones; nearby versions probably work but were not tested.

### Runtime — `requirements-runtime.txt`

| Package | Version | Used by | For |
|---|---|---|---|
| `torch` | 2.11.0 (CUDA 12.8 or CPU build) | `src/model/`, `src/preprocess.py`, `pageocr/engine.py`, `wordocr/` | the model and inference |
| `numpy` | 2.3.5 | throughout | arrays |
| `opencv-contrib-python` | 4.10.0.84 | `pageocr/`, `src/preprocess.py`, `wordocr/fullpage.py` | image decoding, resizing, segmentation |
| `pillow` | 12.3.0 | `wordocr/inputs.py`, `pageocr/imageio.py`, `src/preprocess.py` | reading uploads, transparency, EXIF rotation |
| `streamlit` | 1.65.0 | `app.py`, `app_page.py`, `app_page_auto.py` | the web interface |

Only the standard `cv2` functions are used; the plain `opencv-python` package would probably serve as well, but the contrib build is the one that was tested.

### Training and evaluation — `requirements-training.txt` (adds to the runtime list)

| Package | Version | Used by | For |
|---|---|---|---|
| `PyYAML` | 6.0.3 | `src/train.py`, `tools/` | configuration files |
| `editdistance` | 0.8.1 | `src/metrics.py`, `tools/evaluate.py` | character and word error rates |
| `fonttools` | 4.65.0 | `src/synth/fonts.py` | reading font tables |
| `uharfbuzz` | 0.56.2 | `src/synth/` | shaping Tamil text |
| `freetype-py` | 2.5.1 | `src/synth/render.py` | rasterising glyphs |
| `pyarrow` | 25.0.1 | `tools/import_hf_parquet.py`, `tools/check_split.py` | reading the dataset's Parquet files from disk |

### Tests — `requirements-test.txt` (adds to the training list)

No further Python package: the suites use the standard library's `unittest`, and the synthetic test images are rendered with the generator above. The two browser suites additionally need Node.js and Microsoft Edge.

## Installed with the above, not used directly

Installing the packages above brings their own dependencies (for `streamlit` among others `pandas`, `pyarrow`, `protobuf`, `requests`, `tornado` or `starlette`, `watchdog`; for `torch` among others `sympy`, `networkx`, `jinja2`, `fsspec`). The project's code imports none of them. `requests` in particular is present because the interface framework depends on it; it was not loaded in any audited inference run.

## Removed from the requirements

These were listed in the original project's `requirements.txt` or installed in its environment. No retained file imports them.

| Package | Why it was there | Evidence that it is not needed |
|---|---|---|
| `torchvision` | image transforms in the earlier (legacy) project | imported only by two legacy files, which are not in the release; not loaded in any audited run |
| `fastapi`, `uvicorn`, `python-multipart` | web API of the legacy project | imported only by legacy code |
| `pandas` | data inspection in scratch scripts | imported only by excluded scratch files (it is still installed as a dependency of `streamlit`) |
| `matplotlib` | plots in tools of the real-page pilot | imported only by excluded tools |
| `tqdm` | progress bars | not imported by any retained file |

## Never part of the project environment, and not needed

The following were found elsewhere on the development computer, not in the project's environment. They are listed so that nobody concludes from their absence here that something is missing.

| Item | Status |
|---|---|
| PARSeq model file, `strhub` | no reference in any project file; cannot be loaded into `CRNNv2` |
| Hugging Face Hub client, `transformers`, `tokenizers`, `safetensors`, `datasets`, `sentence-transformers` | not imported; the dataset importer reads local Parquet files with `pyarrow` |
| `google/mt5-small`, IndicBERT | language-model experiments of the legacy project; no reference in current code |
| PaddleOCR, PaddlePaddle, PaddleX, `timm` | not imported |
| Ollama | not referenced |
| Tesseract, EasyOCR, any cloud OCR or API client | not referenced |
| PyMuPDF (`fitz`), `pytest` | used only by the excluded tools and tests of the real-page pilot |

In the isolation test the application produced identical output with all of these unreachable.

## How the lists were derived

1. Every `import` in every retained `.py` file was parsed (`release_tools/dependency_inventory.py`); modules of the standard library and of the project itself were set aside.
2. Each remaining module was mapped to its package and to the areas that import it: runtime (`app*.py`, `wordocr/`, `pageocr/`, `src/` except training and synthesis), training (`src/train.py`, `src/synth/`, `src/augment.py`, `src/metrics.py`, `tools/`), tests.
3. The test suites were run in the release copy; a missing package would fail an import.

Remaining uncertainty: the lists were verified in the development environment, which contains more packages than the lists name. A fresh environment built only from the requirement files has not been created, because nothing was to be installed during this work. A reviewer who builds one is performing that check for the first time; `RELEASE_AUDIT_SUMMARY.md` lists this as a condition.
