# OCR Post-Training Forensic Analysis

## 1. Executive Summary
The `ocr_stabilized` experiment successfully ran for 10 epochs without mode collapse. The model is learning visual representations of Tamil characters. However, a severe architectural bottleneck was discovered during this forensic audit: the Convolutional Neural Network (CNN) downsamples spatial width too aggressively, leaving the sequence decoder (CTC) with insufficient timesteps to output full words. As a result, predictions are systematically truncated, dropping the second half of almost every word.

## 2. Epoch-by-Epoch Training History

| Epoch | Train Loss | Val Loss | CER | Char Acc | WER | Word Acc | Blank % | Avg Pred Len |
|-------|------------|----------|-----|----------|-----|----------|---------|--------------|
| 1     | 3.2394     | 3.0553   | 84.52% | 15.48% | 100.0% | 0.00% | 98.27% | 2.99 |
| 2     | 1.9051     | 2.6727   | 76.32% | 23.68% | 99.97% | 0.03% | 97.97% | 3.51 |
| 3     | 1.6260     | 2.4010   | 63.58% | 36.42% | 99.91% | 0.09% | 97.17% | 4.88 |
| 4     | 1.4527     | 2.2271   | 58.87% | 41.13% | 99.82% | 0.18% | 97.11% | 4.90 |
| 5     | 1.3541     | 2.1054   | 56.68% | 43.32% | 99.67% | 0.33% | 97.16% | 4.83 |
| 6     | 1.2842     | 2.0279   | 54.25% | 45.75% | 99.77% | 0.23% | 97.03% | 5.09 |
| 7     | 1.2265     | 1.9882   | 54.15% | 45.85% | 99.59% | 0.41% | 97.08% | 5.01 |
| 8     | 1.1755     | 1.9460   | 52.28% | 47.72% | 99.64% | 0.36% | 96.89% | 5.32 |
| 9     | 1.1336     | 1.9226   | 51.99% | 48.01% | 99.47% | 0.53% | 96.92% | 5.28 |
| 10    | 1.1082     | 1.9192   | 51.87% | 48.13% | 99.46% | 0.54% | 96.88% | 5.36 |

**Trend Assessment**: Training was healthy. Both losses decreased smoothly. CER improved from 84% to 51%. Character accuracy improved steadily, and prediction length increased from 2.99 to 5.36. 

## 3. Prediction-Length Analysis

*Total samples analyzed: 1632*
- **Mean Ground-Truth Length:** 9.74 characters
- **Mean Predicted Length:** 5.30 characters
- **Mean Input Timesteps (T):** 32.00 (Standardized by batch padding / CNN downsampling)
- **Prediction-to-Target Length Ratio:** 0.54

**Conclusion**: The model is consistently outputting only half of the requested sequence length. This confirms a severe sequence truncation issue.

## 4. Blank Analysis

- **Raw CTC Blank Probability (Timestep Level):** ~93.91%
- **Average Non-Blank Timesteps per sequence:** 5.34
- **Average Decoded Output Length:** 5.30

**Conclusion**: The reported "96.88% Blank Percentage" in the training logs measures the raw timestep blank probability. A value of ~94-96% is completely normal for a CTC model trained on sequences padded to T=32 where the actual valid output is ~5 characters long (5 / 32 = 15.6% non-blanks). The model is not suffering from mode collapse; it is using blanks correctly to separate predictions, but it runs out of temporal resolution to predict the rest of the word.

## 5. Character-Level Analysis

The model is successfully recognizing early-sequence characters but systematically truncating words. 

- **Good Recognition**: Starting characters and root words are often perfect (e.g. `பச்ச`, `முகவ்`, `அணுக்`).
- **Failure Mode**: The model stops emitting characters midway through the word. It is not substituting randomly; it literally hits an architectural wall and stops predicting.

## 6. Word-Level Analysis

Representative samples of Ground Truth vs Prediction:

| Ground Truth | Prediction | Observation |
|--------------|------------|-------------|
| பச்சடிக்குப் | பச்ச் | Dropped `டிக்குப்` |
| மக்களைத் | மக்க் | Dropped `ளைத்` |
| இறந்தனர் | இறந்த் | Dropped `னர்` |
| விநியோகிப்பதில் | விந்தி் | Substituted `நியோகிப்` with `ந்தி` and truncated |
| கம்பெனிகளிடமிருந்து | கம்ப் | Dropped `ெசய்கி்` |
| செல்வகுமார் | கெவ்வ் | Substitution and truncation |
| மணமுடித்து | மந்த் | Truncation |

**Conclusion**: The Word Accuracy is 0.54% because practically 0% of words can be completely predicted before the sequence cuts off.

## 7. Tokenizer / Vocabulary Verification

- **Vocabulary Size**: 78 (1 blank + 77 Tamil characters/symbols)
- **Blank Index**: 0
- **Integrity**: The tokenizer correctly maps Unicode Tamil characters. The "character accuracy" metric is correctly measuring per-token (codepoint/character) accuracy based on the vocabulary mapping. There is no issue with the tokenizer.

## 8. Dataset Label Analysis

- **Min label length**: 2 characters
- **Max label length**: 26 characters
- **Average length**: 9.74 characters

The dataset is predominantly word-level (single words per image crop).

## 9. Real Image Analysis

Using `best.pth` on real images:
- **sample.jpg**: `அட` (Truncated)
- **sample_10lines.png**: Outputs short fragments for every single line (`ச`, `வா்`, `போட்`). 
- **Conclusion**: Real-world images suffer from the exact same temporal truncation.

## 10. Evaluation Script Audit

Audited `scripts/eval_crnn.py`:
- CER (Character Error Rate) and WER (Word Error Rate) are calculated correctly using Levenshtein distance (`editdistance` library).
- Blank percentage calculates the raw probability of the blank token across all timesteps.
- **Verdict**: The evaluation metrics are mathematically sound.

## 11. Baseline Comparison

Because the previous `ocr_stabilized` experiment introduced `BatchNorm2d` to the CRNN encoder, the historical checkpoints (`tamil_full_40epoch`, `fast_track_ctc_fix`) have an incompatible `state_dict` and cannot be loaded directly into the current code without removing BatchNorm. 
However, their training logs reflect the exact same historical symptoms: high CER, ~0% WER, and truncated predictions. The bottleneck has been present since the beginning of the project.

## 12. Determine the Real Bottleneck

**Root Cause: E. Sequence-length problem / C. Model Capacity (CNN Downsampling)**

**Evidence**: 
The CRNN uses standard 2x2 max-pooling layers which halve the spatial width of the image. An input image resized to `height=32` typically has a width of around `128` pixels. 
- 128 / 2 (Pool 1) = 64
- 64 / 2 (Pool 2) = 32
- 32 / 1 (Pool 3, `less_downsample`) = 32
- 32 / 1 (Pool 4) = 32

The final number of timesteps $T$ for CTC is 32. 
CTC requires $T \ge L$, where $L$ is the length of the label. In practice, because CTC requires blank tokens to separate repeating characters, $T$ needs to be roughly $2L + 1$. 
For a 26-character Tamil word, the model needs $T \approx 52$. Because $T=32$, the CTC layer is mathematically incapable of decoding long words, resulting in the model predicting the first few characters and then abruptly collapsing into blanks.

## 13. Decision — NO TRAINING YET

**DECISION: B. CODE/DATA FIX REQUIRED BEFORE TRAINING**

The OCR is absolutely **NOT** ready to freeze. Training it further will not solve the issue, as it is a hard mathematical limit in the CNN architecture's receptive field.

### Exact Expected Changes for Next Experiment
1. **Modify `models/recognition/crnn.py`**:
   - Change the pooling strategy in the CNN encoder. Instead of isotropic `(2, 2)` max pooling, we must use anisotropic `(2, 1)` pooling in the later layers to preserve the spatial width (timesteps) while downsampling height.
2. **Modify `scripts/dataset_loader.py`**:
   - Ensure `AspectRatioPreservingResize` does not artificially cap image widths too aggressively, or adjust the default resizing logic to allow wider sequence generation.

No new training should be started until these architectural modifications are made to the CRNN encoder.
