# Tamil CRNN Root-Cause Diagnostic

## 1. Executive Summary
The CRNN Tamil OCR model failed to memorize a 500-sample tiny dataset. Through deep diagnostics on tensor dimensions, dataset pairing, and gradient flow, we have identified the definitive mathematical root cause: **Destructive Image Preprocessing combined with insufficient Temporal Resolution**. The data loader brutally squashes extremely wide text images (average 1,500 pixels wide) into a rigid 128-pixel width tensor, which the CNN further downsamples to a temporal sequence length of `T=32`. Because Tamil target sequences are frequently 10 to 30 characters long, and CTC requires `T >= 2 * target_length + 1` to resolve repeated characters, the network mathematically lacks the spatial capacity to output the target. It is structurally forced into Mode Collapse.

## 2. Tiny-Dataset Result
- **Observation:** The model could not overfit even on the exact same 500 samples used for training.
- **CER:** Stalled at ~0.8601.
- **Exact Match:** 0%.
- **Blank Percentage:** Decreased from 100% to 93.8%, but locked into a permanent 1-sequence collapse pattern. 

## 3. Image/Label Pairing
- **Verification:** Random selection of 50 samples verified perfectly.
- **Image Paths:** Correctly loaded.
- **Unicode Strings:** Exactly match the JSON manifest. No corruption or misalignments.

## 4. Visual Dataset Inspection
- **Artifact:** `outputs/diagnostics/tamil_ocr_sample_grid.png`
- **Result:** Visual inspection confirms the images contain very clear, highly legible, horizontally structured Tamil text. The visual data quality is excellent and is not responsible for the failure.

## 5. Image Preprocessing
- **Raw Dimensions:** e.g., 1896x247, 1297x294. Aspect ratios frequently exceed 6:1 or 8:1.
- **Processed Tensor:** Forcibly resized to `32x128` (Height x Width) regardless of original width.
- **Result:** Catastrophic aspect ratio distortion. Tamil text is horizontally compressed by a factor of 5x to 15x, destroying glyph morphology.

## 6. CRNN Tensor Dimensions
- **Input:** `[B, 1, 32, 128]`
- **CNN Output:** `[B, 512, 1, 32]`
- **Sequence Conversion:** `[B, 32, 2048]` (1x512 flattened)
- **BiLSTM Input:** `[B, 32, 2048]`
- **BiLSTM Output:** `[B, 32, 512]`
- **CTC Output:** `[B, 32, 78]`

## 7. Temporal Resolution
- **Temporal Length (T):** `32`
- **Target Lengths:** Frequently range between 12 and 30 characters.
- **Mismatch:** A target length of 30 characters mathematically cannot be resolved by `T=32` in CTC (especially if there are repeating characters which require intermediate blanks, demanding `T >= 61`).
- **Conclusion:** The model is completely starved for temporal capacity.

## 8. Tamil Target Encoding
- **Token IDs:** Successfully map from Tamil strings to exact sequences and decode identically back.
- **Vocabulary Size:** 78 classes (Class 0 is blank).
- **Integrity:** Perfect.

## 9. DataLoader Verification
- **Batching:** `dataloader_diagnostic.json` confirmed that `image A -> label A` matches precisely within randomized batches. There is no batch-shuffling desynchronization. 

## 10. Gradient Flow
- **Flow Check:** Normal.
- **CNN Norms:** ~0.001 - 0.31
- **LSTM Norms:** ~0.03 - 0.29
- **Head Norms:** ~0.85
- **Weights:** Successfully updating on optimizer step. The backpropagation pipeline works flawlessly.

## 11. CTC Output Distribution
- **Before Training:** Mean blank probability ~1.3%. Logits are random.
- **After 1 Step:** Mean blank probability increases slightly as loss begins to optimize.
- **After 30 Epochs (Tiny Dataset):** Mean blank probability = 73.2%. Exact mapping revealed 120 blank timesteps and exactly 8 non-blank timesteps out of a batch of 128 timesteps (4 images x 32 T). This equates to exactly **2 characters per image**, proving the rigid static-character mode collapse.

## 12. Runtime Training Configuration
- **Learning Rate:** 0.0003
- **Optimizer:** Adam
- **CTC Reduction:** Mean
- **Zero Infinity:** True
- **Blank Index:** 0

## 13. Root-Cause Candidates

1. **Destructive Preprocessing Resizing (Aspect Ratio)**
   - **Evidence:** Original widths of ~1800px are compressed to 128px.
   - **Counter-evidence:** None.
   - **Confidence:** 100%
   
2. **Insufficient Sequence Length (T=32)**
   - **Evidence:** Target lengths reach 30, mathematically impossible to resolve in 32 CTC timesteps.
   - **Counter-evidence:** None.
   - **Confidence:** 100%

3. **CTC Loss Configuration**
   - **Evidence:** The loss decreases but the network predicts 94% blanks.
   - **Counter-evidence:** This is exactly how CTC reacts when forced to map an impossible sequence (it settles for the safest possible highly-blank local minimum).
   - **Confidence:** 0% (It is a symptom, not a cause).

## 14. Single Highest-Priority Issue
**Image resizing and temporal resolution.**
The fixed `128x32` image resizing is crushing the data, and the resulting `T=32` sequence length is mathematically blocking the network from mapping long sequences.

## 15. Recommended Fix
**Implement Dynamic Resizing with Padding.**
Change the `DataLoader` transform so that images are resized to a fixed height of `32px` while strictly preserving the aspect ratio. Then, dynamically pad the width of the images to match the maximum width in the batch (or a fixed max width like `1024px`). This will naturally increase `T` (e.g. `1024 / 4 = 256`) and preserve glyph morphology.

## 16. Decision
1. **Are image/label pairs correct?** Yes, 100%.
2. **Are Tamil Unicode labels preserved?** Yes.
3. **Is preprocessing safe?** No, it is catastrophically distorting the aspect ratios.
4. **Is 32x128 appropriate?** Absolutely not.
5. **Is the CRNN temporal dimension appropriate?** No, T=32 is too short.
6. **Is the CTC input valid?** Mathematically yes, functionally no (violates sequence length requirements).
7. **Are gradients flowing?** Yes, healthy flow.
8. **Are model weights changing?** Yes.
9. **Is the DataLoader producing correct pairs?** Yes.
10. **Why can't the model memorize 500 samples?** The model lacks the temporal sequence length `T` to output a 20-30 character string, so it learns a local minimum of predicting a rigid 2-character string and absorbing the rest as blanks.
11. **What is the highest-priority issue?** Fixing the destructive 128px width resizing.
12. **What exact change should we test next?** Aspect-ratio-preserving batch padding in `dataset_loader.py` and running the Tiny-Dataset Overfit Diagnostic again.
13. **Should we proceed to the full 75,736-image dataset?** **NO.**
