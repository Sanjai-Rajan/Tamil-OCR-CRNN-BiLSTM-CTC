# OCR Model Comparison

## 1. Objective
This document provides a direct head-to-head comparison of the final quantitative and qualitative evaluations for the two most recent models trained in this repository:
1. `ocr_temporal_fix` (Verified model with standard schedule)
2. `ocr_balanced_temporal` (New model with OneCycleLR and slightly modified epoch configurations)

Both models successfully use the critical `less_downsample=True` architectural change (T=W/8) implemented to mitigate the initial sequence truncation failure.

## 2. Architecture Difference
Both models share the **exact same architecture**:
- Backbone: CNN (CRNN-style)
- Sequence: BiLSTM (Input 2048)
- Decoder: CTC
- Pooling: `(2,1)` for the last two stages to preserve temporal width (`less_downsample=True`), yielding a temporal resolution of `W/8`.

**The difference lies strictly in the training hyperparameter trajectory:**
- `ocr_temporal_fix` utilized a standard schedule, completing smoothly and generalizing well on test and real-world inputs.
- `ocr_balanced_temporal` utilized a `OneCycleLR` scheduler with mixed precision (`bfloat16`) and was halted at Epoch 10 exactly on the scheduler's completion step.

## 3. Overall TEST Metrics Comparison
Evaluation on the full, identical native TEST split (43,110 samples).

| Metric | `ocr_temporal_fix` | `ocr_balanced_temporal` | Difference |
| :--- | :--- | :--- | :--- |
| **TEST Samples** | 43,110 | 43,110 | 0 |
| **CER** | 0.2376 | **0.2176** | **-0.020** (Better) |
| **WER** | 0.4311 | **0.4199** | **-0.0112** (Better) |
| **Character Accuracy** | 76.24% | **78.24%** | **+2.00%** (Better) |
| **Word Accuracy** | 56.89% | **58.01%** | **+1.12%** (Better) |
| **Exact Word Matches** | 24,527 | **25,007** | **+480** (Better) |
| **Blank Percentage** | 97.92% | 92.97% | -4.95% (Lower) |
| **Avg Target Length** | 4.79 tokens | 4.79 tokens | Same |
| **Avg Pred Length** | 3.04 tokens | 3.26 tokens | +0.22 tokens |

*Observation:* On the tightly cropped TEST dataset bounding boxes, `ocr_balanced_temporal` actually edges out the previous model, proving the `OneCycleLR` schedule yielded a slightly more precise statistical model for character alignment.

## 4. Length-Bucket Metrics

### Bucket 1: 1–5 tokens (28,594 samples)
| Model | CER | Exact Word Accuracy | Avg Pred Length |
| :--- | :--- | :--- | :--- |
| `ocr_temporal_fix` | 0.0789 | 85.76% | 1.97 |
| `ocr_balanced_temporal` | **0.0696** | **87.30%** | 1.99 |

### Bucket 2: 6–10 tokens (8,594 samples)
| Model | CER | Exact Word Accuracy | Avg Pred Length |
| :--- | :--- | :--- | :--- |
| `ocr_temporal_fix` | 0.4938 | 0.07% | 4.99 |
| `ocr_balanced_temporal` | **0.4534** | **0.52%** | 5.53 |

### Bucket 3: 11–15 tokens (4,777 samples)
| Model | CER | Exact Word Accuracy | Avg Pred Length |
| :--- | :--- | :--- | :--- |
| `ocr_temporal_fix` | 0.6145 | 0.00% | 5.32 |
| `ocr_balanced_temporal` | **0.5720** | 0.00% | 5.98 |

### Bucket 4: 16+ tokens (1,145 samples)
| Model | CER | Exact Word Accuracy | Avg Pred Length |
| :--- | :--- | :--- | :--- |
| `ocr_temporal_fix` | 0.7061 | 0.00% | 5.50 |
| `ocr_balanced_temporal` | **0.6663** | 0.00% | 6.38 |

## 5. Behavior Analysis

### 5.1 Prediction-Length Behavior
- **`ocr_temporal_fix`** capped its prediction length around ~5.5 tokens for long words.
- **`ocr_balanced_temporal`** pushes this ceiling slightly higher, averaging 6.38 tokens for the longest bucket. This explains the lower CER in longer buckets, as it outputs slightly more characters before mathematically stopping due to the CTC frame limit constraint.

### 5.2 Blank Behavior
- **`ocr_temporal_fix`** emitted blanks 97.92% of the time, leading to slightly shorter but highly confident predictions.
- **`ocr_balanced_temporal`** emitted blanks 92.97% of the time. This massive reduction in blanks correlates with the known mode-collapse observed during its validation loss (the model frequently defaults to emitting `்` or single characters repeatedly rather than remaining confidently blank). While this "guessing" behavior improved statistical metrics on the test set, it creates unstable visual outputs on real-world data.

### 5.3 Long-Word Truncation
Neither model solves long-word truncation for >6 tokens. The truncation is a mathematical certainty induced by resizing images to a small fixed standard width before passing them into the CNN, capping the temporal frames (W/8) fed into the CTC decoder below the character count of long Tamil words.

## 6. Trade-offs and Winner

- **Quantitative Winner (TEST dataset):** `ocr_balanced_temporal` wins across all statistical metrics (CER, WER, Exact Matches), pushing accuracy ~2% higher on short words.
- **Qualitative/Robustness Winner (Real-world images):** `ocr_temporal_fix` is significantly more stable. `ocr_balanced_temporal` suffers from character mode-collapse when faced with uncropped/unpadded real-world images (e.g., repeating meaningless suffixes), whereas `ocr_temporal_fix` fails gracefully by truncating text cleanly.

**Recommendation:** The integration backend (`app/inference.py`) should use `ocr_balanced_temporal` ONLY if the bounding boxes exactly match the training set padding conditions. For general-purpose robust scanning in production, `ocr_temporal_fix` remains the safer, more stable deployment choice despite the 2% statistical deficit.
