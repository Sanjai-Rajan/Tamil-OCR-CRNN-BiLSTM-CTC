> **Note for this repository.** This document was written for the project's private 2026-10-10 review release and is kept unchanged below this note. Differences here: trained weights (`runs/*/best.pth`) are **not included** (see `docs/MODEL_CHECKPOINTS.md`); the release evidence files (`RELEASE_*`, `RELEASE_SHA256SUMS.txt`), `release_tools/`, `release_evidence/`, `GLOBAL_CLEANUP_PLAN.md` and `CHANGELOG_CLEAN_RELEASE.md` are not part of this repository; the other release documents now live in `docs/`. The repository's own record is `EXPORT_MANIFEST.csv` and `REPOSITORY_AUDIT.md`.

# Installation and Use

## Requirements

| | Tested | Notes |
|---|---|---|
| Python | 3.12.10 (64-bit) | 3.12 is the supported version. Other versions were not tested. |
| Operating system | Windows 11 | The code uses no Windows-only feature; Linux and macOS were not tested. |
| GPU | NVIDIA GPU with the CUDA 12.8 build of PyTorch | Optional. Without a GPU the application runs on the CPU, more slowly. |
| Disk | about 550 MB | of which 530 MB are the four checkpoints |
| Internet | only to install the packages | The application itself makes no network request. |

No account, token, API key or model download is needed.

## 1. Create an environment

In the folder of this release:

```
python -m venv .venv
.venv\Scripts\activate
```

On Linux or macOS the second line is `source .venv/bin/activate`.

## 2. Install PyTorch

Choose one line.

```
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

The first is the tested GPU build (CUDA 12.8); the second is CPU only.

## 3. Install the other packages

To run the application:

```
pip install -r requirements-runtime.txt
```

To also train, evaluate or run the tests, use `requirements-training.txt` or `requirements-test.txt` instead; each includes the one before it. `DEPENDENCIES.md` explains every package.

## 4. Check the checkpoint

The default model file must be at `runs/experiment_002a/best.pth`. Its SHA-256 must be

```
fb29131ed71fe0c89229497d0171b1ff2de4e1e581b4d8c1e09d6fbbaf62d5f1
```

Check it yourself:

```
certutil -hashfile runs\experiment_002a\best.pth SHA256
```

(`sha256sum runs/experiment_002a/best.pth` on Linux or macOS.) The application performs the same check every time it loads the model and refuses a file with any other hash; it never substitutes another file. `RELEASE_SHA256SUMS.txt` lists the hash of every file of the release.

## 5. Start the application

```
python -m streamlit run app.py
```

Open `http://localhost:8501`. Stop with Ctrl+C. The server listens on this computer only, and usage statistics are switched off (`.streamlit/config.toml`).

### Word OCR

1. Open the *Single image* tab and upload an image of **one word** (PNG, JPEG, TIFF or WebP), or open the *Batch* tab and upload several such images. In a batch each file is read separately, and a file that fails does not stop the others.
2. Press *Run OCR*.
3. The result can be downloaded as TXT, JSON or CSV.

An image that looks like a line or a page is refused here with a pointer to the *Full-page OCR* tab.

### Full-page OCR

1. Open the *Full-page OCR* tab.
2. Upload an image of one page (PNG, JPEG, TIFF or WebP; not PDF).
3. The page is segmented and the detected word regions are drawn on it. The segmentation settings can be adjusted.
4. Press *Read page*. Each region is read as a word, and the text is assembled in reading order.
5. The result can be downloaded as TXT, JSON or CSV, with the box of every word.

The *About and limits* tab states what the application does and does not do.

### The other two interfaces

`python -m streamlit run app_page.py` (you draw or import the word boxes) and `python -m streamlit run app_page_auto.py` (automatic segmentation with more settings) are earlier interfaces kept for completeness. `app.py` is the supported one.

## 6. Run the tests

```
pip install -r requirements-test.txt
python -B run_all_tests.py --no-browser
```

All test inputs are generated in memory; no dataset is needed and nothing is downloaded. Expected: five suites, all `PASS`.

`python -B run_all_tests.py` also runs two browser suites. They need Node.js and Microsoft Edge, start their own local server on ports 8765 and 8781, and stop it again. If either tool is missing, those two suites are reported as `NOT RUN`.

`python -B run_all_tests.py --list` shows the suites without running them.

## Optional: protect additional folders

The application refuses the project's protected research files by fingerprint (`SECURITY_AND_PRIVACY.md`). If you keep such files on your computer and want files added later to be refused too, name their folders in the environment variable `TAMIL_OCR_PROTECTED_ROOTS` (separated by `;` on Windows, `:` elsewhere) before starting the application. A folder named there that does not exist makes the application refuse all uploads until the setting is corrected. Most users do not need this variable.

## Troubleshooting

| Symptom | Cause |
|---|---|
| "…is not the registered checkpoint…" | the checkpoint file is damaged or was replaced; restore it and compare the hash |
| "The research-data guard cannot check this file…" | `pageocr/protected_manifest.json` is missing or altered, or `TAMIL_OCR_PROTECTED_ROOTS` names a missing folder |
| "The image has at most two gray levels…" | the image is pure black and white; use the original grayscale or colour scan |
| Tests are skipped with a message about rendering libraries | `uharfbuzz`, `freetype-py` or `fonttools` is not installed; install `requirements-test.txt` |
