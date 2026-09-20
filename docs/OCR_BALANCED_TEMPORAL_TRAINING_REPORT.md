# OCR Balanced Temporal Training Report

## 1. Experiment Overview
The `ocr_balanced_temporal` experiment was continued for 5 additional epochs within the available 5-hour training window. The training successfully resumed from Epoch 5 and reached Epoch 10, correctly restoring model architecture, optimizer state, and RNG state. The scheduler was allowed to complete its configured 10-epoch lifespan, successfully saving checkpoints at the end of each epoch.

## 2. Completed Training Metrics
Metrics at the end of Epoch 10 (validation split):
- **Epochs Completed**: 10 (Resumed at 5, completed 10)
- **Train Loss**: 1.0524
- **Validation Loss**: 1.8618
- **CER (Character Error Rate)**: 0.5034 (50.34%)
- **WER (Word Error Rate)**: 0.9908 (99.08%)
- **Character Accuracy**: 0.4966 (49.66%)
- **Word Accuracy**: 0.0092 (0.92%)
- **Blank Percentage**: 92.31%
- **Average Prediction Length**: 5.66 tokens

## 3. Comparison Against `ocr_temporal_fix`
| Metric | `ocr_temporal_fix` (Test) | `ocr_balanced_temporal` (Val Epoch 10) | Difference |
| :--- | :--- | :--- | :--- |
| **CER** | 0.2376 | 0.5034 | **+0.2658** (Worse) |
| **Character Acc.** | 0.7624 | 0.4966 | **-0.2658** (Worse) |
| **WER** | 0.4311 | 0.9908 | **+0.5597** (Worse) |
| **Word Acc.** | 0.5689 | 0.0092 | **-0.5597** (Worse) |
| **Avg. Prediction Len.**| 3.04 | 5.66 | **+2.62** (Longer) |
| **Blank Percentage** | 97.92% | 92.31% | **-5.61%** (Fewer blanks) |

## 4. Long-Word Truncation Analysis
Did long-word truncation improve? **Yes, but at the cost of accuracy.**

The average prediction length increased significantly from **3.04 tokens** to **5.66 tokens**. The blank percentage also dropped from 97.92% to 92.31%, indicating that the model is actively predicting more characters per image and utilizing the temporal space better.

However, the dramatic drop in Character Accuracy (76.24% → 49.66%) and Word Accuracy (56.89% → 0.92%) suggests that while the model is outputting longer sequences to overcome truncation, it is struggling to predict the *correct* characters. The output has essentially become longer, but much noisier. The model seems to be suffering from a form of mode collapse, predicting a high frequency of specific dominant characters (e.g., `'்'` accounts for 26.68% of predictions). 

## 5. Checkpoint Paths
The training loop correctly generated the following updated checkpoints:
- **Best Model Checkpoint (Lowest CER = 0.5034):**
  `checkpoints\recognition\tamil\ocr_balanced_temporal\best.pth`
- **Latest Checkpoint (End of Epoch 10):**
  `checkpoints\recognition\tamil\ocr_balanced_temporal\latest.pth`
- **Pre-Validation Checkpoint (Epoch 10):**
  `checkpoints\recognition\tamil\ocr_balanced_temporal\epoch_010_pre_validation.pth`

*(Note: The training correctly halted at the beginning of Epoch 11 as the `OneCycleLR` scheduler hit its maximum allocated steps of 39,510, safely concluding the scheduled training block.)*
