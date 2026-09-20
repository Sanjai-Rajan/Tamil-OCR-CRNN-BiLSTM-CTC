# Tamil OCR for Historical Manuscripts using CRNN-BiLSTM-CTC

This repository contains the source code, training procedures, and diagnostic evidence for an Optical Character Recognition (OCR) system designed to recognize printed Tamil scripts and historical manuscripts. 

## 🚨 Model Weights Excluded
**All neural models used in this project were trained from scratch. No pretrained OCR model, pretrained transformer, pretrained language model, or pretrained neural-network weights are included.**

**The trained production checkpoint and the large training datasets are intentionally excluded from this GitHub repository.** 
The production model (`checkpoints/recognition/tamil/ocr_balanced_temporal/best.pth`) is a massive PyTorch checkpoint and is managed separately. Therefore, this repository cannot perform out-of-the-box trained OCR inference without the trained weights. It serves as an open architectural reference and reproducibility archive.

## Architecture
The system employs a standard standard **CRNN-BiLSTM-CTC** architecture:
- **CNN Feature Extraction**: A standard convolutional backbone extracts deep visual features from the normalized text-line images.
- **BiLSTM Sequence Modelling**: Bidirectional Long Short-Term Memory networks capture temporal context across the horizontal sequence.
- **CTC Decoding**: Connectionist Temporal Classification aligns the unsegmented feature sequences to the target Tamil character sequences.

## Pipeline & Features
- **Segmentation/Reconstruction**: Advanced Computer Vision (`models/cv/segmenter.py`) isolates character sequences.
- **Deterministic Post-OCR Correction**: A deterministic conceptual corrector (`models/digitalization/conceptual_corrector.py`) utilizes a corpus-derived Tamil vocabulary (`tamil_correction_vocabulary.json`) to correct minor OCR permutations safely.
- **Training Methodology**: Training scripts operate on large balanced datasets, processing variable-length sequences. The methodology accounts for complex Tamil ligatures.

## Experiments & Diagnostics
- **Baseline Benchmark**: The W/4 baseline experiment establishes performance bounds.
- **CTC Temporal-Capacity Diagnostic**: Detailed investigations into length-based performance degradation, specifically analyzing why the temporal resolution constraints affect longer word sequences.
- **W/2 Temporal-Resolution Experiment**: Explores structural changes to the pooling layers to double the temporal output capacity and resolve long-sequence failures.

## Project Structure
- `app/`: FastAPI frontend application structure and inference bindings.
- `models/`: PyTorch neural network definitions.
- `scripts/`: Dataset loaders, training scripts, and evaluation suites.
- `data/`: Contains the vocabulary mappings and small text corpus fragments used for determinisitic correction (The massive image datasets are excluded).
- `outputs/`: Training histories, diagnostic plots, and benchmark results.

## Requirements
See `requirements.txt`. Requires Python 3.12+ and PyTorch.
