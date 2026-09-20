# Phase 2: Refinement Baseline

All subsequent controlled experiments are measured against this verified baseline model.

## Baseline Checkpoint Information
- **Checkpoint Path:** `outputs/training/tamil/tamil_full_40epoch/best.pth` (Stored in `checkpoints/recognition/tamil/tamil_full_40epoch/best.pth`)
- **Baseline Epoch:** 80

## Dataset Configuration
- **Dataset Size:** ~126,419 training samples (17 packets)
- **Vocabulary Size:** 77 unique characters + 1 CTC Blank (Total 78 classes)

## Training Configuration
- **Architecture:** CRNN + CTC (6-layer CNN, 2-layer BiLSTM 256-hidden, Linear Head)
- **Batch Size:** 32 (Inferred from previous run configs)
- **Learning Rate:** Variable (typically `3e-4` to `1e-4` during original tuning)

## Evaluated Baseline Metrics (Validation Set)
- **CER:** 0.3584
- **WER:** 0.8144
- **Character Accuracy:** 0.6610 (66.10%)
- **Word Accuracy:** 0.1855 (18.55%)
- **Blank Percentage:** 86.54% (User reported baseline calculation), Verified diagnostically as 75.2% raw timestep blank probability.
