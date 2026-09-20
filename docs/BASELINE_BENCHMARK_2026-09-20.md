# Current Production OCR Baseline Benchmark

## Checkpoint
- **Path:** `checkpoints\recognition\tamil\ocr_balanced_temporal\best.pth`
- **SHA-256 Hash:** `07554040440E80E5EC2B1BD6EA1FA5BB6618E105AE3AD45F7590F21113E2F136`

## Dataset
- **Split:** Native `test` split (spanning 5 packets).
- **Total Samples:** 43,110
- **Vocabulary Path:** `data\tamil_ocr_dataset\vocabulary\tamil_vocab.json`

## Evaluation Configuration
- **Date/Time:** 2026-09-20T09:51:35+05:30
- **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU
- **Frameworks:** Python 3.12.10, PyTorch 2.11.0+cu128
- **Evaluation Script:** `scripts\eval_test_balanced.py`
- **Batch Size:** 128
- **Post-Correction:** STRICTLY DISABLED. This evaluates the raw CRNN outputs via standard CTC decode.

## Overall Results
- **CER (Character Error Rate):** 0.2176 (21.76%)
- **WER (Word Error Rate):** 0.4199 (41.99%)
- **Character Accuracy:** 78.24%
- **Word Accuracy:** 58.01%
- **Exact Matches:** 25,007
- **Average Prediction Length:** 3.26
- **Average Target Length:** 4.79
- **Blank Percentage:** 92.97%

## Length-wise Results

### Target Length 1–5
- **Samples:** 28,594
- **Average Target Length:** 1.97
- **Average Prediction Length:** 1.99
- **CER:** 0.0696 (6.96%)
- **Exact Accuracy:** 87.30%

### Target Length 6–10
- **Samples:** 8,594
- **Average Target Length:** 8.13
- **Average Prediction Length:** 5.53
- **CER:** 0.4534 (45.34%)
- **Exact Accuracy:** 0.52%

### Target Length 11–15
- **Samples:** 4,777
- **Average Target Length:** 12.57
- **Average Prediction Length:** 5.98
- **CER:** 0.5720 (57.20%)
- **Exact Accuracy:** 0.00%

### Target Length 16+
- **Samples:** 1,145
- **Average Target Length:** 17.92
- **Average Prediction Length:** 6.38
- **CER:** 0.6663 (66.63%)
- **Exact Accuracy:** 0.00%

## Comparison With Previously Documented Benchmark
| Metric | Previous | Fresh | Difference |
| :--- | :--- | :--- | :--- |
| **CER** | 21.76% | 21.76% | 0.00 percentage points |
| **WER** | 41.99% | 41.99% | 0.00 percentage points |
| **Character Accuracy** | 78.24% | 78.24% | 0.00 percentage points |
| **Word Accuracy** | 58.01% | 58.01% | 0.00 percentage points |
| **Exact Matches** | 25,007 | 25,007 | 0 difference |
| **Target Length** | 4.79 | 4.79 | 0.00 difference |
| **Prediction Length** | 3.26 | 3.26 | 0.00 difference |

## Interpretation
The freshly measured baseline metrics are mathematically identical to the historical reference up to 4 decimal places. The CRNN suffers from an aggressive sequence-truncation failure mode on medium-to-long words (Target Length > 5), where the average prediction length collapses (e.g., predicting 5.53 characters for an 8.13 character target). This is purely a raw OCR architectural behavior, isolated completely from any downstream conceptual correction. 

## Integrity Verification
- **Checkpoint Hash Before:** `07554040440E80E5EC2B1BD6EA1FA5BB6618E105AE3AD45F7590F21113E2F136`
- **Checkpoint Hash After:** `07554040440E80E5EC2B1BD6EA1FA5BB6618E105AE3AD45F7590F21113E2F136` (Identical)
- **Source Code / Dataset:** Unmodified during evaluation.
