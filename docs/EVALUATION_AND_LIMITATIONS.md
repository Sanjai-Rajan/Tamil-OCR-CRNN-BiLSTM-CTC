> **Note for this repository.** This document was written for the project's private 2026-10-10 review release and is kept unchanged below this note. Differences here: trained weights (`runs/*/best.pth`) are **not included** (see `docs/MODEL_CHECKPOINTS.md`); the release evidence files (`RELEASE_*`, `RELEASE_SHA256SUMS.txt`), `release_tools/`, `release_evidence/`, `GLOBAL_CLEANUP_PLAN.md` and `CHANGELOG_CLEAN_RELEASE.md` are not part of this repository; the other release documents now live in `docs/`. The repository's own record is `EXPORT_MANIFEST.csv` and `REPOSITORY_AUDIT.md`.

# Evaluation and Limitations

## Summary

- The model has been measured on **two kinds of word images**: the test split of the handwriting dataset it was trained on, and synthetic printed words made by the project's own generator.
- It has **not** been measured on real printed documents in a way that supports an accuracy claim.
- Full-page OCR adds a segmentation step that has been tested only on generated pages.

## Results that exist

All figures are for **single word images**, greedy decoding, no dictionary. CER is the character error rate, WER the share of words with at least one error. The records are in `reports/experiment_002a_final_evaluation.md` (included, unmodified) and, for Experiment 002-B, in the original project's evaluation records.

### Handwriting: IIIT-INDIC-HW-WORDS Tamil, test split (16,184 word images)

| Model | CER | WER |
|---|---|---|
| Baseline (handwriting only) | 0.60 % | 4.46 % |
| Experiment 001 | 0.75 % | 5.54 % |
| **Experiment 002-A (default)** | **0.61 %** | **4.52 %** |
| Experiment 002-B | 0.69 % | 5.03 % |

### Synthetic print: 6,000 generated word images (3,000 modern style, 3,000 old style)

| Model | CER, all | CER, modern | CER, old | WER, all |
|---|---|---|---|---|
| Baseline | 19.16 % | 18.51 % | 19.82 % | 47.35 % |
| Experiment 001 | 0.26 % | 0.11 % | 0.42 % | 1.78 % |
| **Experiment 002-A (default)** | **0.17 %** | **0.06 %** | **0.27 %** | **1.10 %** |

### Comparisons made (paired bootstrap, 10,000 resamples, 95 % interval)

- 002-A against 001 on the handwriting test: CER lower by 0.14 percentage points [0.11, 0.18].
- 002-A against the baseline on the handwriting test: CER +0.01 points [−0.02, +0.04], within the pre-set tolerance of +0.10.
- 002-A against 002-B on the handwriting test: 002-A lower by 0.08 points [0.05, 0.11].

## How these numbers must be read

1. **The handwriting test is mostly seen vocabulary.** 85.3 % of the test labels also occur among the training labels, and six test images are byte-identical to training images. The test measures recognition of words much like those seen in training, in the handwriting of this dataset. It does not measure reading unknown words.
2. **The synthetic test is made by the same generator, fonts and word list as the synthetic training data.** It shows that the model learned the generator's distribution. It is not evidence about scanned books or newspapers.
3. **One training run per experiment.** The intervals above cover sampling of test items, not the variation between training runs with different seeds.
4. **The differences between 001, 002-A and 002-B are confounded.** Batch composition, effective batch size and number of steps changed together.
5. **Checkpoint selection used the handwriting validation split.** The test split was scored once, after training.

## Real documents

The project ran a pilot on real printed pages. Its reference transcriptions were produced by an AI assistant and were **not verified by a person**, so its scores are not reported here as accuracy. A second evaluation on development pages is blocked until a human review that has not been carried out. Neither data set is part of this release.

**Unverified, therefore not claimed:** any accuracy figure for real printed Tamil documents; performance on handwriting outside the IIIT dataset; performance on historical print; robustness to photographs, curved pages, stamps, tables or mixed scripts.

## Full-page OCR

Segmentation is classical image processing: background correction, Sauvola thresholding, connected components, skew correction up to 5°, column gutters, line bands, word gaps, reading order. It has been tested on generated pages only (one and two columns, headings, small skew, noise).

On those generated pages, and on the 6 pages of the release comparison, the text is reproduced exactly. That is a statement about clean generated pages in fonts the model knows. On real pages expect segmentation errors: merged or split words, lines broken by ornaments, wrong column order in complex layouts, text lost near figures.

## What the application does not do

- No PDF input.
- No dictionary, language model or spelling correction: the output is exactly what the network emitted.
- No space character: the model reads one word; spaces and line breaks in page text come from the geometry of the detected regions.
- No calibrated confidence: the score shown is the mean of the winning class probabilities, useful only to decide which results to look at first.
- Only the 100 characters of `src/charset.py` (Tamil letters and signs, Tamil and Latin digits, common punctuation). Other characters cannot be output.
- Images with only two gray levels (binarised scans) are refused: the model was trained on grayscale images.
- Words much longer than the training range are clipped to the maximum input width; the result notes when this happens.
- Only exact copies of the project's protected research files are refused by the guard (`SECURITY_AND_PRIVACY.md`).

## What the audits and tests of this release show, and do not show

| Shown | Not shown |
|---|---|
| The release produces exactly the same output as the original project on 32 synthetic inputs | accuracy on any real document |
| Output is independent of external model caches, credentials, other Python packages and network, on the tested paths | behaviour of code paths that were not run |
| The checkpoints descend from the project's own random initialisation | which data each run saw, beyond what configurations and logs record |
| The test suites pass on generated inputs | that the tests cover every input a user may supply |

## Reproducing the evaluation

With the dataset installed as described in `TRAINING_GUIDE.md`:

```
python tools/evaluate.py --ckpt runs/experiment_002a/best.pth --manifest data/real/iiit_hw/manifest_test.jsonl --samples --out <output prefix>
```

The handwriting figures should be reproducible to within GPU rounding. The synthetic figures depend on the fixed synthetic validation set, which is not included and can only be regenerated with the original word list.
