# Contributing

Thank you for your interest. **No licence has been chosen for this project yet** (`LICENSE_SELECTION_REQUIRED.md`). Please ask the maintainer before contributing, so that the terms of your contribution are clear.

## Ground rules

- **Never commit model weights, datasets, real document images, annotations, credentials or `.env` files.** `.gitignore` covers the usual cases; check `git status` before every commit anyway.
- **Keep the recognizer local.** No external OCR engine, cloud API, pretrained-model download or telemetry.
- **New image-processing or segmentation behaviour goes behind an opt-in option** (see `exp_rule_removal`, `exp_contrast_clahe`, `exp_crop_rule_fill`, `exp_angle_lines`), off by default. A default changes only with evidence, ideally from real pages with human-verified references.
- **Never use model output as ground truth.** References come from people or from a generator's own labels.
- **Report only measured results**, and say what was not tested.
- **Do not weaken a test to make it pass.** If behaviour changes on purpose, change the test exactly and say why.

## Development setup

```
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -r requirements-test.txt
python -B run_all_tests.py --no-browser
```

Without a registered checkpoint, the model-dependent suites are reported as NOT RUN (`docs/MODEL_CHECKPOINTS.md`). The browser suites need Node.js and Microsoft Edge.

## Experiments

Use `dev_tools/segmentation_experiments.py` or `dev_tools/slant_experiments.py` with a new `--out` folder:

- fix the decision rule before running;
- keep development and held-out splits separate, and never tune on held-out pages;
- outputs are not committed (`.gitignore`); summarise them in `docs/EXPERIMENTS.md`.
