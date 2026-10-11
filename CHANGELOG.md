# Changelog

## Repository export (2026-10-11)

First public-repository form of Tamil OCR V2, exported from the project's diagnostic development workspace of 2026-10-10 (the latest verified state). There is no earlier Git history; the private history is not included.

**Changed for this repository** (the source code is otherwise identical to the workspace; see `EXPORT_MANIFEST.csv`):
- trained weights, datasets, experiment outputs, logs and private release/workspace tooling are left out;
- the release documents moved to `docs/`, each with a note on what differs here;
- new: `README.md`, `.gitignore`, `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, `LICENSE_SELECTION_REQUIRED.md`, `docs/MODEL_CHECKPOINTS.md`, `docs/EXPERIMENTS.md`, `runs/README.md`, `EXPORT_MANIFEST.csv`, `REPOSITORY_AUDIT.md`;
- `run_all_tests.py`: when the registered default checkpoint is missing, the suites that need it are reported as NOT RUN, naming the file, instead of failing (`REPOSITORY_AUDIT.md`).

## State of the code at export

- **Word OCR and full-page OCR** in `app.py`. Recognizer: CRNNv2 with CTC; segmentation: classical (`pageocr/`).
- **Diagnostic mode** (opt-in): overlays, region table, crop inspection, heuristic warnings, evidence bundle, manual-vs-pipeline crop comparison, and angle-aware line boxes with a horizontal-band comparison.
- **Experimental segmentation options, all off by default:**
  - `exp_rule_removal`: candidate on generated pages;
  - `exp_contrast_clahe`: rejected;
  - `exp_crop_rule_fill`: candidate, changes model input;
  - `exp_angle_lines`: rejected.

  Evidence: `docs/EXPERIMENTS.md`.
- Tests: 16 unit-test modules (329 tests at export, with the registered checkpoint present) and 3 browser smoke suites.
