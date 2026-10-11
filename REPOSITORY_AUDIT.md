# Repository audit — Tamil_OCR_Github (2026-10-11)

What was exported, what was checked, and what is still open before publication. Every number below comes from a command that was actually run.

## 1. Source

**Source:** the diagnostic development workspace of 2026-10-10 (last changed 20:26 that day). It is the newest Tamil OCR V2 folder and the only one with the latest verified implementation:
- the review release;
- diagnostic mode, rule removal, the segmentation experiments, and the angle-aware line geometry and experiment.

**Workspace verification before the export:**
- full test run 329/329 unit tests, 0 skipped, plus 3 browser suites (31, 23 and 42 checks);
- default OCR output identical to the record made before any workspace change (32/32 outputs, 480/480 regions);
- clean integrity check.

**Other folders in the same parent directory:**
- `tamil_ocr_v2`: the original project, older; contains datasets, protected material and legacy code;
- `tamil_ocr_v2_review_2026-10-10`: an older review copy, and its `.rar` archive;
- `1_Draft`, last changed 2026-09-22.

None was used as a source. None was modified.

**Chain:** workspace → `Tamil_OCR_Sourcecode` (a clean archive, byte-for-byte copies) → `Tamil_OCR_Github` (this repository), produced by `Tamil_OCR_Sourcecode/_export_tools/build_exports.py`. Both are independent folders (no links). No Git history was copied; this repository was initialised fresh.

## 2. What is included, what is excluded

`EXPORT_MANIFEST.csv` lists every file the export considered: path, category, size, SHA-256, status and reason. Summary:

| | Files |
|---|---|
| Files of the source-code archive considered | 220 |
| Copied (160 in total), of which unchanged in place | 153 |
| Release documents moved to `docs/` with a short note on top | 6 |
| Copied and then changed for the repository | 1 (`run_all_tests.py`) |
| Written for the repository | 11 (`README.md`, `.gitignore`, `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, `LICENSE_SELECTION_REQUIRED.md`, `docs/MODEL_CHECKPOINTS.md`, `docs/EXPERIMENTS.md`, `runs/README.md`, this file, the manifest) |
| Excluded | 60 |

**Excluded from the repository** (all are kept in `Tamil_OCR_Sourcecode`, except weights, which neither folder contains):
- private release evidence and tooling: `RELEASE_*` (6), `release_tools/` (9), `CHANGELOG_CLEAN_RELEASE.md`;
- the old `README.md`, replaced;
- workspace session records and result summaries: `reports/diagnostic_implementation/**` (25), summarised in `docs/EXPERIMENTS.md`;
- training histories and console logs: `runs/*/history.*`, `train.log` (12). The four frozen `runs/*/config.yaml` files are included;
- six workspace-only dev tools (`create_workspace.py`, two `patch_*.py`, `final_integrity.py`, `verify_experiments.py`, `default_output_compare.py`), which need private folders or records.

**Excluded already from the source-code archive** (`Tamil_OCR_Sourcecode/EXPORT_MANIFEST.csv`, 948 files):
- the 4 checkpoint binaries `runs/*/best.pth`, about 530 MB;
- 867 generated experiment page and overlay images;
- 13 browser-test screenshots and downloads;
- 33 generated fixture images (`diagnostic_fixtures/`);
- raw experiment and inference records, test and experiment console logs (which contain absolute paths of the development computer), workspace copy manifests;
- `GLOBAL_CLEANUP_PLAN.md` (machine-specific).

**Never present in the workspace, so never copied:** datasets (`data/`, `external/`), the word list, protected research material (`real_docs/`, `development_pages/`, the R1 review folders), legacy code and checkpoints, Hugging Face caches, the PARSeq model, credentials, virtual environments.

## 3. Changes made for the repository (and nowhere else)

1. **`run_all_tests.py`.** If the registered default checkpoint is missing, the 8 test modules that load it, and the 3 browser suites, are reported as **NOT RUN**, naming the file; every other module runs unchanged. With the checkpoint present, it behaves exactly as in the workspace. No test was edited, skipped inside a test, or weakened.
   - The 8 modules were determined by measurement: each was run alone without the checkpoint.
     - 40 of the 55 failure tracebacks name the checkpoint or the unloaded model.
     - 14 fail because the app disables its Run buttons when no model is loaded (confirmed separately: section 5).
     - 1 was a missing `runs/<id>/config.yaml`, fixed by including those files.
2. **Release documents** moved to `docs/` with a note on top; their text is otherwise unchanged.
3. **New repository documents**, listed above.

The code is otherwise identical to the workspace: 152 code, configuration and font files compared byte for byte; the only difference is `run_all_tests.py`. **No change was made to the original project, the review copy or the workspace.**

## 4. Security scan of the repository

A pattern scanner (`Tamil_OCR_Sourcecode/_export_tools/` holds the export tools; the scanner itself was a session script) read every text file. It reports file and line only, never values.

| Check | Result |
|---|---|
| Credential patterns: AWS, GitHub, Hugging Face, OpenAI, Anthropic, Slack and Google keys, private-key blocks, JWTs, `api_key / secret / password / token = "..."` | 0 |
| `.env`, `.git-credentials`, `.netrc`, `.pypirc`, key or certificate files | 0 |
| Model weights (`.pth .pt .ckpt .onnx .safetensors .bin .h5 .pb .tflite .gguf .pkl .pickle .joblib`) | 0 |
| Nested `.git`, `.venv`/`venv`/`env`, `node_modules`, `__pycache__`, `.cache`, `.huggingface` | 0 |
| Bytecode (`.pyc`) | 0 |
| Personal absolute paths (`X:\Users\…`, `/home/…`, `/Users/…`, the developer's user name, `AppData`) | 0 |
| E-mail addresses | 7, all in `assets/fonts/OFL_*.txt` line 1: the font authors' copyright notices, which the OFL requires to keep |
| Images, PDFs | 0 |
| Files over 1 MB | 0 |
| Python files that do not parse | 0 |

The scanner was **self-tested**: planted fake values (a Hugging Face-style token, an AWS-style key, a private-key header, hard-coded secrets, Windows, Linux and macOS home paths, a `.pth`, a `.env`, a `.venv`) were all detected. The same scan of `Tamil_OCR_Sourcecode` found only the font e-mails, and 5 personal paths in the historical workspace reports, which are kept in the local archive only.

**Protected material:**
- `pageocr/protected_manifest.json` holds sizes and SHA-256 hashes of protected research files: no names, paths or content.
- Code and tests name the protected folders (`real_docs`, `development_pages`, `R1_*_review`) only to refuse them.
- No research image, annotation or document is included.

**External services:**
- No Python file imports `huggingface_hub`, `transformers`, `tokenizers`, `safetensors`, `datasets`, `openai`, `anthropic`, `requests`, a cloud SDK or a third-party OCR engine.
- The only URLs are `localhost` (the browser tests) and the PyTorch package index in installation comments.
- `tools/import_hf_parquet.py` reads local Parquet files with `pyarrow`; it downloads nothing and needs no login.
- Watch-lists naming such libraries (`release_tools/`) are not in the repository.

## 5. Dependencies and imports

**Every third-party import is covered by the requirements files, and every pinned package is imported somewhere:**

| File | Packages | Used for |
|---|---|---|
| `requirements-runtime.txt` | `torch`, `numpy`, `opencv-contrib-python`, `pillow`, `streamlit` | model, images, segmentation, the app |
| `requirements-training.txt` | + `PyYAML`, `editdistance`, `fonttools`, `uharfbuzz`, `freetype-py`, `pyarrow` | configs, metrics, the HarfBuzz renderer; `pyarrow` only for two offline dataset tools |
| `requirements-test.txt` | = the training set | tests render synthetic Tamil |

- No package was added or removed.
- No `opencv-contrib` module is used, so `opencv-python` would probably suffice; this was **not tested**, so the tested pin was kept.
- No `torchvision`, `transformers` or Hugging Face package is required.

**Startup without a checkpoint** (Streamlit's `AppTest` on this repository's `app.py`):
- no exception;
- the message "Model not available … not found at runs/experiment_002a/best.pth … it is not downloaded automatically";
- *Run OCR* (2 buttons) and *Read page* disabled.

That segmentation still runs after a page upload without a model is established from the code (`app.py` segments independently of the model), not from an app test.

## 6. Tests run in this repository (no checkpoint, local interpreter)

`python -B run_all_tests.py`, with the project's existing Python 3.12 environment:

| Suite | Result |
|---|---|
| guard | 14 / 14 PASS |
| pageocr (boxes, V3 geometry) | 43 / 43 PASS |
| diagnostics (`test_slant_lines`) | 10 / 10 PASS |
| training_code | 33 / 33 PASS |
| 8 modules that load the checkpoint: `test_pageocr_engine`, `test_pageocr_v3_segmentation`, `test_wordocr_app`, `test_wordocr_service`, `test_fullpage_app`, `test_fullpage_ocr`, `test_diagnostics`, `test_line_geometry` | NOT RUN (checkpoint not distributed) |
| 3 browser suites | NOT RUN (they drive the real model) |

**Total:** 100 tests passed, 0 failed, 0 skipped. Runner exit code 2: not everything could run. These counts cover the repository as it would be cloned.

With the checkpoint present, the identical code passed 329/329 unit tests and 31/23/42 browser checks in the workspace on 2026-10-10 (only `run_all_tests.py` differs, and only when the checkpoint is missing).

## 7. Git

- `git init` (branch `main`).
- No remote, no commit, no push.
- `git status --ignored` after the final `.gitignore`: **all repository files visible, none ignored.** The first `.gitignore` hid `src/data/` through a `data/` rule; the dataset rules are now anchored to the root.
- `git check-ignore` confirms that weights anywhere (`*.pth`), `runs/*/best.pth`, `/data/`, `/real_docs/`, `.env`, `/diagnostic_runs/` and experiment outputs under `reports/` are ignored.

## 8. Not verified

- **No clean environment was created.** Installation from the requirements files on a fresh machine, and Linux or macOS, were not tested; all tests used the project's existing environment.
- The model-dependent tests were not run in this repository (no checkpoint). They passed in the workspace on identical code.
- Real-document accuracy is not established (`docs/EVALUATION_AND_LIMITATIONS.md`, `docs/EXPERIMENTS.md`).
- That the fonts are unmodified upstream releases was not re-checked against their original sources.

## 9. Before publishing: decisions and risks

1. **Licence.** None chosen; see `LICENSE_SELECTION_REQUIRED.md`. Also confirm that no institutional (for example C-DAC) or programme terms restrict publication.
2. **Protected-data fingerprints** (`pageocr/protected_manifest.json`): confirm that publishing the hashes of the protected files is acceptable. Removing the file makes the application refuse every upload.
3. **Unsafe loading in two tools.** `src/train.py` (resume) and `tools/evaluate.py` use `torch.load(weights_only=False)`; documented in `SECURITY.md`. The application itself uses `weights_only=True`.
4. **Training data terms.** The IIIT handwriting dataset's terms bear on any future release of weights; weights are not in this repository.
5. **Document provenance.** The documents in `docs/` were written for the private review release. They mention the original project's history, for example "two checkpoints of unverified origin" in the legacy material and audit procedures. Review whether that level of detail should be public.
6. **Credentials outside the project.** None was found in the project. Any token that ever existed on the development computer, for example a Hugging Face login, is outside this repository; if it was ever exposed elsewhere, revoke it at its issuer (deleting a file does not revoke it).
7. **Before the first commit,** run `git status` and confirm that no `*.pth`, data or image file appears.
