# Model card — Tamil OCR V2 word recognizer (CRNNv2, Experiment 002-A)

## Summary

| | |
|---|---|
| Task | recognise the text of **one cropped image of a single Tamil word** (printed or handwritten) |
| Architecture | CRNNv2: residual CNN (64-128-256-384) → 2-layer BiLSTM (256) → linear, CTC loss; 11,038,277 parameters (`src/model/crnn_v2.py`) |
| Input / output | grayscale word crop, scaled to height 48 px (width 16–1024 px) → a character sequence over 100 characters (`src/charset.py`), greedy CTC decoding |
| Default checkpoint | `runs/experiment_002a/best.pth`, SHA-256 `fb29131ed71fe0c89229497d0171b1ff2de4e1e581b4d8c1e09d6fbbaf62d5f1`, about 127 MB |
| Training | from scratch (seed 42), 40 epochs, IIIT-INDIC-HW-WORDS Tamil handwriting + synthetic printed words 50/50, paired-source batches (`configs/experiment_002a.yaml`) |
| Pretrained weights used | **none**. The seed-initialisation test (`docs/MODEL_PROVENANCE.md`) shows that every weight tensor descends from the project's own seed-42 random initialisation |
| Developer | the project owner, with substantial AI-assisted tooling and documentation (`docs/MODEL_PROVENANCE.md`) |

Three further registered checkpoints (Baseline, 001, 002-B) exist for the documented comparisons (`wordocr/checkpoints.py`, `docs/MODEL_PROVENANCE.md`).

## Availability of the weights

**The weights are not stored in this Git repository:**
- each file (about 127 MB) exceeds GitHub's 100 MB per-file limit;
- their redistribution depends on the terms of the third-party training data (IIIT-INDIC-HW-WORDS publishes none) and of the word list;
- the owner has not yet decided whether to publish them, and how: Git LFS or a release asset. See `DEPLOYMENT.md`, step 1.

To use the model, place an authorised copy at the registered path and run `python tools/verify_checkpoints.py`. The application:
- loads only registered files with the registered hash;
- uses `torch.load(weights_only=True)`;
- never downloads a model and never substitutes another file.

## Intended use

- Reading **single-word crops** of modern printed or handwritten Tamil, locally, as a research and development tool.
- The word recognizer inside the app's full-page OCR, after the classical segmenter has cut the page into words.

**Out of scope, or not validated:**
- whole lines or pages given directly to the model;
- historical print;
- scripts other than Tamil;
- characters outside the 100-character set;
- any use where errors have consequences without a person checking the output.

## Evaluation evidence

All figures are for **single word images**, greedy decoding, no dictionary. Details and intervals: [docs/EVALUATION_AND_LIMITATIONS.md](docs/EVALUATION_AND_LIMITATIONS.md); record: `reports/experiment_002a_final_evaluation.md`.

| Test set | CER | WER |
|---|---|---|
| IIIT-INDIC-HW-WORDS Tamil, test split (16,184 images) | 0.61 % | 4.52 % |
| Synthetic print, project generator (6,000 images) | 0.17 % | 1.10 % |

**How to read these:**
- 85.3 % of the handwriting test labels occur in training, and six test images are byte-identical to training images: mostly seen vocabulary, one handwriting distribution.
- The synthetic test comes from the same generator, fonts and word list as the synthetic training data.
- There was one training run per experiment.
- **No accuracy on real documents has been established.** The real-page pilot's references were produced by an AI assistant and not verified by a person, so they are not used as evidence.

## Limitations and risks

- **Unknown real-world accuracy.** Expect errors on unseen handwriting styles, unusual fonts, low resolution, photographs and degraded scans.
- **No language model or dictionary:** the output is exactly what the network emitted. The displayed score is **not** a calibrated probability.
- **No space character:** page text spacing comes from the segmenter.
- **Segmentation dominates page errors** on the tested generated pages, notably notebook ruling ([docs/EVALUATION_AND_LIMITATIONS.md](docs/EVALUATION_AND_LIMITATIONS.md)).
- **Dataset bias:** the handwriting comes from one dataset's writers; the synthetic print from 15 open fonts.
- **Unsafe loading in two tools:** `src/train.py --resume` and `tools/evaluate.py` load checkpoints with `weights_only=False`. Use them only with trusted files.

## Reproducing

- **Evaluation:** with the dataset prepared ([DATASETS.md](DATASETS.md)):
  `python tools/evaluate.py --ckpt runs/experiment_002a/best.pth --manifest data/real/iiit_hw/manifest_test.jsonl --samples --out <prefix>`
- **Training:** [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md). Not reproducible from this repository alone: it needs the handwriting dataset and the original word list. Bit-identical weights are not expected even with the same data (GPU non-determinism).
