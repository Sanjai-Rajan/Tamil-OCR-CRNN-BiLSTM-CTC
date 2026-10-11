> **Note for this repository.** This document was written for the project's private 2026-10-10 review release and is kept unchanged below this note. Differences here: trained weights (`runs/*/best.pth`) are **not included** (see `docs/MODEL_CHECKPOINTS.md`); the release evidence files (`RELEASE_*`, `RELEASE_SHA256SUMS.txt`), `release_tools/`, `release_evidence/`, `GLOBAL_CLEANUP_PLAN.md` and `CHANGELOG_CLEAN_RELEASE.md` are not part of this repository; the other release documents now live in `docs/`. The repository's own record is `EXPORT_MANIFEST.csv` and `REPOSITORY_AUDIT.md`.

# Training Guide

How the four documented checkpoints were trained, what you need to train again, and what this package can and cannot reproduce.

## Can training be reproduced from this package?

**Not from the package alone.** The code, configurations and fonts are here. The data is not:

| Needed | Included? | How to get it | Needed for |
|---|---|---|---|
| Training code, model, generator, configurations | yes | — | all experiments |
| Fonts for the synthetic generator | yes (`assets/fonts/`, 29 font files, SIL Open Font License) | — | 001, 002-A, 002-B |
| IIIT-INDIC-HW-WORDS, Tamil (handwritten word images) | **no** — third-party dataset, about 5 GB | from its source, see below | all experiments |
| Word list for the synthetic generator (`tamil_correction_vocabulary.json`, 522,495 word forms with frequencies) | **no** — built from Tamil text collections (Project Madurai and others) whose redistribution terms were not established | not obtainable from this package; a substitute can be built, see below | 001, 002-A, 002-B |
| Synthetic validation set (6,000 generated images) | **no** — generated; depends on the word list | regenerate with `tools/synth_export.py` | checkpoint reports of 001, 002-A, 002-B |

Consequences, stated plainly:

- **Baseline** (handwriting only) needs only the IIIT dataset and can be retrained with what is documented here.
- **Experiments 001, 002-A, 002-B** additionally need the word list. With a substitute word list you train *a* model of the same kind, not the same model.
- Even with identical data, **bit-identical weights are not expected**: GPU training with mixed precision is not deterministic across hardware and driver versions. What can be checked exactly is the starting point: `torch.manual_seed(42)` followed by `CRNNv2(101)` reproduces the initial weights of every run (`MODEL_PROVENANCE.md`).
- Nobody has yet retrained these models from this package on a second machine. Treat the instructions below as documented procedure, not as a verified reproduction.

## Data

### IIIT-INDIC-HW-WORDS (Tamil)

Handwritten word images with labels, by Santhoshini Gongidi and C. V. Jawahar, CVIT, IIIT Hyderabad (`https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words`). The project used the Parquet copy published as the Hugging Face dataset `c3rl/IIIT-INDIC-HW-WORDS-Tamil`: 75,736 training, 11,598 validation and 16,184 test images. Check the dataset's terms of use for your purpose; this project did not verify them for redistribution, which is why the data is not included.

Convert the Parquet files to the project's format (image files plus one `manifest_<split>.jsonl` per split). The tool reads local files only:

```
python tools/import_hf_parquet.py --src <folder with the parquet files> --inspect
python tools/import_hf_parquet.py --src <folder with the parquet files> --out data/real/iiit_hw --name iiit_hw
python tools/check_split.py --help
```

Image bytes are written unchanged. Labels are Unicode-normalised (NFC); labels with characters outside the 100-character set are left out and listed in `import_report.txt`.

Known properties of this dataset, relevant to how results are read: 85.3 % of the test labels also occur among the training labels, and six test images are byte-identical to training images.

### Synthetic printed words

Rendered during training by `src/synth/`: a word is drawn from the word list, shaped with HarfBuzz, rasterised with FreeType in one of the fonts of `assets/fonts/`, and degraded to look like modern or old print (`src/synth/degrade.py`). Nothing is stored; each epoch sees new images.

A substitute word list is a JSON object `{"word": frequency, ...}`. `tools/build_wordlist.py` builds one from any folder of Tamil text files you have the right to use, or from a Tamil Wikipedia dump:

```
python tools/build_wordlist.py --txt-dir <folder of .txt files> --out data/text_corpus/my_words.json
```

Then put its path in the `vocab:` line of the configuration.

`tools/pretrain_check.py` is the project's pre-training check. One of its assertions is that the original word list is present at `data/text_corpus/tamil_correction_vocabulary.json` with SHA-256 `3f19ac2db418d3061162acfd978463b3f3a0c21652e4968c8383fa2e24a2241a`. Without that file, or with a substitute, this assertion fails and the check exits with code 1; that is expected in this package and concerns only that assertion.

Fixed synthetic validation set, as used for the reports:

```
python tools/synth_export.py --vocab <word list> --out data/synth_val --n-modern 3000 --n-old 3000
```

## Configurations

`configs/`, and identical copies in each `runs/<experiment>/config.yaml`.

| | Baseline | 001 | 002-A | 002-B |
|---|---|---|---|---|
| File | `baseline_hw.yaml` | `experiment_001.yaml` | `experiment_002a.yaml` | `experiment_002b.yaml` |
| Sources | IIIT handwriting | IIIT + synthetic print, 50/50 | as 001 | as 001 |
| Batch composition | — | mixed freely | **paired**: every batch drawn from both sources | mixed freely |
| Batch size | 64 | 64 | 64 (realised mean 44) | 44 |
| Samples per epoch | 75,736 | 151,472 | 151,472 | 151,472 |
| Epochs | 40 | 40 | 40 | 40 |
| Seed | 42 | 42 | 42 | 42 |

Common to all: learning rate 0.001 with one-cycle schedule and 5 % warm-up, AdamW with weight decay 0.01, mixed precision, gradient clipping at 5.0, batches of similar-width images, dropout 0.25. Model: channels 64-128-256-384, LSTM hidden size 256, two layers, input height 48.

`configs/mixed_print_hw.yaml` is an early draft with unfilled values. It was never run and is included only because it is part of the original configuration folder.

## Commands

```
pip install -r requirements-training.txt
python tools/pretrain_check.py --config configs/baseline_hw.yaml
python -m src.train --config configs/baseline_hw.yaml
```

Output goes to `runs/<experiment>/`: `best.pth` (lowest validation character error rate), `last.pth`, `history.csv`, `history.json`, `config.yaml`, `train.log`, and per-epoch sample predictions.

**Training writes into `runs/<experiment>/`.** To keep the shipped checkpoints intact, change the `experiment:` name in a copy of the configuration before training. `--resume` continues an interrupted run from its own `last.pth`; it is the only way the training script loads weights.

Evaluation on a manifest, and paired comparison of two models:

```
python tools/evaluate.py --ckpt runs/<experiment>/best.pth --manifest data/real/iiit_hw/manifest_test.jsonl --samples --out <output prefix>
python tools/compare_eval.py --help
```

## What the original runs looked like

From the `train.log` and `history.csv` files in `runs/`:

| | Baseline | 001 | 002-A | 002-B |
|---|---|---|---|---|
| Duration | 5.7 h | 10.9 h | 9.6 h | 12.0 h |
| Mean training loss, first → last epoch | 2.87 → 0.005 | 1.71 → 0.008 | 1.60 → 0.008 | 1.58 → 0.007 |
| IIIT validation CER after the first epoch | 13.6 % | 9.6 % | 5.7 % | 9.4 % |
| Epoch saved as `best.pth` | 36 | 31 | 35 | 39 |
| IIIT validation CER at that epoch | 0.57 % | 0.72 % | 0.61 % | 0.65 % |

One local NVIDIA GPU; peak GPU memory about 6–7 GB; about 160–200 images per second.

## A note on loading checkpoints

`src/train.py` (resume) and `tools/evaluate.py` load checkpoint files with `torch.load(..., weights_only=False)`, which can execute code contained in a file. They were left as they are so that the training code in this release is exactly the code that produced the checkpoints. Only pass them checkpoint files you trust; the files in this release also load with `weights_only=True`. The application itself always uses the safe mode and a hash check.
