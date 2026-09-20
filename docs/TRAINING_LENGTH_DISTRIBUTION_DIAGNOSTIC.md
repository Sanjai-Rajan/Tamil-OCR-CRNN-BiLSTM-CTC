# Experiment 3A: Training Length Distribution Diagnostic

## 1. Objective
Analyze the exact training set (native 	amil/train split) to determine the distribution of target token lengths and assess whether a short-sequence bias exists in the data that could contribute to the length collapse/saturation observed during CRNN inference.

## 2. Exact Dataset Source
The data was read directly from the production training split:
data/tamil_ocr_dataset/imported/tamil/train
This was performed by parsing the manifest.jsonl files without modifying any data.

## 3. Exact Tokenizer Used
The standard production tokenizer models.digitalization.tokenizer.Tokenizer using data/tamil_ocr_dataset/vocabulary/tamil_vocab.json. This ensures target lengths reflect exactly the number of tokens the CTC head is trained to output, rather than raw Unicode points.

## 4. Total Training Samples
**126,419** valid samples.

## 5. Length-Frequency Summary
- **Overall Average Target Length**: 6.59 tokens
- **Maximum Target Length**: 32 tokens
- **>= 6 tokens**: 53.86%
- **>= 11 tokens**: 22.44%
- **>= 16 tokens**: 4.53%
- **>= 20 tokens**: 1.00%
- **>= 25 tokens**: 0.11%

## 6. Bucket Statistics

| Bucket | Sample Count | Percentage | Avg Target | Median Target | Max Target | Avg Image Width | Avg Width/Token |
|--------|--------------|------------|------------|---------------|------------|-----------------|-----------------|
| 1-5 | 58,322 | 46.13% | 2.14 | 2.0 | 5 | 310.26 px | 120.78 px |
| 6-10 | 39,723 | 31.42% | 8.09 | 8.0 | 10 | 1517.06 px | 192.72 px |
| 11-15 | 22,649 | 17.92% | 12.53 | 12.0 | 15 | 1588.12 px | 127.92 px |
| 16+ | 5,725 | 4.53% | 18.10 | 17.0 | 32 | 1679.43 px | 94.20 px |
| 20+ | 1,271 | 1.01% | 21.90 | 21.0 | 32 | 1703.58 px | 78.54 px |

## 7. Width/Length Statistics
As observed, the average width per token varies significantly by length bucket. Short sequences (1-5 tokens) average 310 px wide, while mid-to-long sequences are aggressively bounded or padded to ~1500-1700 px (the dataset maximum width constraint), causing the pixels-per-token density to plummet from ~192 px/token for 6-10 lengths down to ~78 px/token for 20+ lengths. 

## 8. Representative Examples
Representative examples from varying length buckets have been logged to:
outputs/diagnostics/training_length_distribution/representative_samples.csv

## 9. Interpretation

**A. Is the training dataset dominated by short words?**
Yes. Nearly half of the training set (46.13%) consists of very short sequences (1-5 tokens).

**B-E. Percentages:**
- 1-5 tokens: 46.13%
- 6-10 tokens: 31.42%
- 11-15 tokens: 17.92%
- 16+ tokens: 4.53%

**F-H. Extreme lengths:**
- >= 20 tokens: 1,271 samples (1.01%)
- >= 25 tokens: 147 samples (0.11%)
- Longest target: 32 tokens

**I. Does the training distribution provide substantial exposure to long sequences?**
No. The exposure to sequences longer than 15 tokens is minimal (4.53%), and exposure to extremely long sequences (>= 20) is statistically negligible (1.01%). 

**J. Could a short-sequence training bias plausibly contribute to the ~5–6 token prediction saturation observed in Experiment 2?**
Yes. The measured distribution is highly consistent with a possible short-sequence training bias. The CRNN is predominantly rewarded for predicting short sequences. Furthermore, because image widths plateau aggressively after ~1500 pixels, longer sequences are forced into the same horizontal temporal capacity as 6-10 token sequences, exacerbating the problem as the feature-map per token gets crushed.

## 10. Limitations
- This is a diagnostic measurement of the dataset; it does not prove causality for the length collapse on its own. The results of the W/2 ablation (Experiment 3) will help isolate whether the limitation is purely dataset-driven or architecture-driven (temporal capacity).
- We analyzed target lengths as encoded by the tokenizer, not bounding box widths per character.
