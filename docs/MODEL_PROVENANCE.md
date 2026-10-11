> **Note for this repository.** This document was written for the project's private 2026-10-10 review release and is kept unchanged below this note. Differences here: trained weights (`runs/*/best.pth`) are **not included** (see `docs/MODEL_CHECKPOINTS.md`); the release evidence files (`RELEASE_*`, `RELEASE_SHA256SUMS.txt`), `release_tools/`, `release_evidence/`, `GLOBAL_CLEANUP_PLAN.md` and `CHANGELOG_CLEAN_RELEASE.md` are not part of this repository; the other release documents now live in `docs/`. The repository's own record is `EXPORT_MANIFEST.csv` and `REPOSITORY_AUDIT.md`.

# Model Provenance

What the model is, where its weights come from, and how far that has been checked.

## The model

`CRNNv2` (`src/model/crnn_v2.py`): a residual convolutional network, a two-layer bidirectional LSTM and a linear output layer, trained with CTC loss. 11,038,277 parameters, 101 output classes (100 characters of `src/charset.py` plus the CTC blank), input height 48 pixels. Decoding is greedy; there is no dictionary, language model or correction step.

The architecture is the project's own code and imports only PyTorch. No pretrained backbone is used and the code contains no call that could load one.

## Checkpoints in this release

All four are byte-identical copies of the files in the original project. The application refuses any file whose SHA-256 differs from the registered value (`wordocr/checkpoints.py`, `pageocr/engine.py`).

| Checkpoint | Experiment | Saved at epoch | SHA-256 |
|---|---|---|---|
| `runs/experiment_002a/best.pth` (**default**) | 002-A: IIIT handwriting + synthetic print, 50/50, paired-source batches | 35 | `fb29131ed71fe0c89229497d0171b1ff2de4e1e581b4d8c1e09d6fbbaf62d5f1` |
| `runs/experiment_002b/best.pth` | 002-B: as 002-A, batch size 44, unpaired batches | 39 | `a9db26f91e1ff07d46b9bebe4b8d60154c2ddea11c976f811010b9a2eace1677` |
| `runs/experiment_001/best.pth` | 001: IIIT handwriting + synthetic print, 50/50 | 31 | `d095b2e84daea177bb35d5ae386e1670e6ca3c27af685d9491f5ced08e086f2b` |
| `runs/baseline_hw/best.pth` | Baseline: IIIT handwriting only | 36 | `0b2655176054793ce6cfc7a41f7a90eefd8f19320c30fa3b6a7b302127b5fe1b` |

Where the hashes were first recorded:

- Baseline, 001 and 002-A: `reports/experiment_002a_final_evaluation.md`, section 2. That file is included here unmodified.
- 002-B: the original project's `reports/experiment_002b_evaluation/evaluation_provenance.json`. That record contains absolute paths of the owner's computer and is not included; this document restates its hash.

Each run folder also holds the exact configuration (`config.yaml`), the per-epoch history (`history.csv`, `history.json`) and the training log (`train.log`).

**Not included:** the last-epoch states (`last.pth`), a batch-norm diagnostic derived from Experiment 001, and two checkpoints of an earlier, different project (`archive/checkpoints/` in the original). The last two were excluded because their provenance has not been established; they are not used by any code.

## How the weights were produced

`python -m src.train --config configs/<experiment>.yaml`:

1. the random generators are seeded with the configuration's `seed` (42 in every configuration);
2. `CRNNv2` is constructed with PyTorch's default random initialisation;
3. training runs for 40 epochs (AdamW, one-cycle learning-rate schedule, mixed precision);
4. the epoch with the lowest validation character error rate on the IIIT validation set is saved as `best.pth`.

The only code that loads weights during training is `--resume`, which continues a run from its own `last.pth`. No log of the four runs contains a resume.

Training data is described in `TRAINING_GUIDE.md`. In short: a public handwriting dataset and synthetic images rendered by this project. **Third-party data was used; third-party model weights were not.**

## What has been checked, and by which audit

Two audits were carried out on 2026-10-10 in the original project. Their full evidence stays there; this is a summary.

### Model provenance audit

- **Seed-initialisation test.** Because training seeds the generator and then builds the model, the starting weights of each run can be regenerated. Every one of the 29 weight tensors of each of the four checkpoints is still correlated with the regenerated seed-42 initialisation, at least 12.8 standard deviations above the level given by 40 other seeds. A model initialised from pretrained or foreign weights would show no such relation. *This is the evidence that the checkpoints were trained from the project's own random initialisation.* It can be repeated with the files in this release: regenerate `torch.manual_seed(42); CRNNv2(101)` and compare.
- **Independence of the experiments.** Checkpoints of different experiments are almost uncorrelated with each other apart from the shared starting point, so none was fine-tuned from another.
- **Records.** Embedded configuration, history, logs and file dates agree with each other and with the registered hashes.
- **Search.** No secret in 3,289 text files; no import of any model-hub, OCR-engine or API library in current code.

### Isolation and dependency test

The production inference functions were run on 32 synthetic inputs (26 word images, 6 pages) twice: normally, and in a process with an empty home folder, empty model caches, credential variables removed, external model caches and the second Python installation's packages blocked by path, and all non-local network refused.

- 37 of 37 comparisons identical (26 words, 6 pages, 5 runs through `app.py`), 480 of 480 word regions identical in text and box, score difference 0.0.
- The only model file opened: `runs/experiment_002a/best.pth`.
- No attempt to reach anything but the local computer, in either run.

For this release the comparison was repeated between the original project and this copy; see `RELEASE_AUDIT_SUMMARY.md`.

## What these checks do not establish

- They cover the paths that were run: model loading, Word OCR and Full-page OCR through `app.py`, with the default checkpoint on a GPU, on synthetic inputs. The region apps (`app_page.py`, `app_page_auto.py`) and the training tools were examined by reading the source, not by an isolation run.
- History cannot be proven cryptographically. Logs and dates are ordinary files; the seed-initialisation test is the part that does not depend on them.
- File tracing saw what goes through Python. No operating-system tracer or packet capture was used.
- Both audits were carried out by the AI assistant that also wrote much of the application code. They were designed so that someone else can repeat them, but they are not a third-party audit.
- Good recognition of synthetic images says nothing about accuracy on real documents; see `EVALUATION_AND_LIMITATIONS.md`.

## Use of AI assistance in the project

Not part of the model, but part of an honest provenance statement: much of the tooling, frontend and documentation was written with an AI assistant (Claude), and the reference annotations of the project's real-page pilot were produced by the same assistant, not by a person. Neither enters training or inference. No AI service is called at run time.
