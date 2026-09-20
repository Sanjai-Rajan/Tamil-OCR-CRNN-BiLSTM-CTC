# OCR Balanced Temporal — Final TEST Evaluation

## 1. Objective
This report details the complete, native TEST set evaluation (43,110 samples) for the `ocr_balanced_temporal` experiment using `best.pth` (Epoch 10). 

## 2. Checkpoint and Configuration
- **Path:** `checkpoints\recognition\tamil\ocr_balanced_temporal\best.pth`
- **Architecture:** CRNN with `less_downsample=True` (Temporal resolution W/8)
- **Epoch:** 10

## 3. Overall TEST Metrics
| Metric | Value |
| :--- | :--- |
| **TEST Samples** | 43,110 |
| **CER (Character Error Rate)** | 0.2176 (21.76%) |
| **WER (Word Error Rate)** | 0.4199 (41.99%) |
| **Character Accuracy** | 78.24% |
| **Word Accuracy** | 58.01% |
| **Exact Word Matches** | 25,007 |
| **Blank Percentage** | 92.97% |
| **Average Target Length** | 4.79 tokens |
| **Average Prediction Length** | 3.26 tokens |

## 4. Length-Bucket Analysis

Samples were segmented into buckets based on their ground-truth character/token length to isolate the long-word truncation behavior.

### Bucket 1: 1–5 tokens
- **Sample Count:** 28,594
- **Average Target Length:** 1.97
- **Average Prediction Length:** 1.99
- **CER:** 0.0696 (6.96%)
- **Exact Word Accuracy:** 87.30%

### Bucket 2: 6–10 tokens
- **Sample Count:** 8,594
- **Average Target Length:** 8.13
- **Average Prediction Length:** 5.53
- **CER:** 0.4534 (45.34%)
- **Exact Word Accuracy:** 0.52%

### Bucket 3: 11–15 tokens
- **Sample Count:** 4,777
- **Average Target Length:** 12.57
- **Average Prediction Length:** 5.98
- **CER:** 0.5720 (57.20%)
- **Exact Word Accuracy:** 0.00%

### Bucket 4: 16+ tokens
- **Sample Count:** 1,145
- **Average Target Length:** 17.92
- **Average Prediction Length:** 6.38
- **CER:** 0.6663 (66.63%)
- **Exact Word Accuracy:** 0.00%

## 5. Conclusion
The `ocr_balanced_temporal` checkpoint shows very high accuracy (87.3%) for short words (1-5 tokens). However, like the preceding `ocr_temporal_fix`, it exhibits a hard mathematical ceiling on sequence length dictated by the input width resizing pipeline. The average prediction length for words longer than 15 characters plateaus around 6.38 tokens, confirming that the long-word truncation issue remains bounded by the `W` dimension limit passed from the dataloader.
