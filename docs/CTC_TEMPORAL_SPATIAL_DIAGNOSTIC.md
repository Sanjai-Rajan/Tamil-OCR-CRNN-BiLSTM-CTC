# CTC Temporal Capacity & Spatial Alignment Diagnostic

## Objective
The objective of this diagnostic is to investigate the "Length Collapse" failure mode observed in the baseline CRNN OCR model for long Tamil words. Specifically, we investigate whether the network physically runs out of temporal CTC timesteps ($T$) for longer words (e.g. $T < \text{required\_}T$), or if the model simply fails to utilize available temporal capacity, collapsing into blank-dominant representations.

## Diagnostic Configuration
- **Checkpoint:** `checkpoints\recognition\tamil\ocr_balanced_temporal\best.pth`
- **SHA-256 Before:** `07554040440E80E5EC2B1BD6EA1FA5BB6618E105AE3AD45F7590F21113E2F136`
- **SHA-256 After:** `07554040440E80E5EC2B1BD6EA1FA5BB6618E105AE3AD45F7590F21113E2F136` (Unmodified)
- **Dataset:** `data\tamil_ocr_dataset\imported\tamil\test`
- **Samples Analyzed:** 200 (Deterministically sampled across 4 target-length buckets using seed 42)
- **Post-Correction:** STRICTLY DISABLED.

## Methodology
The CRNN feature extraction downsamples the input width $W$ by a factor of 4 (due to `MaxPool2d(2,2)`, `MaxPool2d(2,2)`, `MaxPool2d(2,1)`). Thus, the number of CTC timesteps is $T \approx W / 4$.
For each sample, we measured:
1. Ground-truth target token length.
2. The minimum required timesteps: $\text{Required\_}T = \text{Target Length} + \text{Number of adjacent identical tokens}$.
3. Actual available capacity ($T$).
4. Temporal margin ($T - \text{Required\_}T$).
5. Decoded predicted length.
6. Raw CTC blank probability percentage before decoding.

## Measured Results

| Bucket | Samples | Avg Target Len | Avg Pred Len | Pred/Target Ratio | Avg $T$ | Avg Required $T$ | Avg Temporal Margin | Avg Blank % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1-5** | 50 | 4.26 | 5.00 | 1.19 | 43.3 | 4.26 | +39.04 | 87.47% |
| **6-10** | 50 | 8.10 | 5.20 | 0.66 | 38.46 | 8.14 | +30.32 | 85.72% |
| **11-15** | 50 | 12.76 | 5.84 | 0.46 | 42.00 | 12.80 | +29.20 | 84.82% |
| **16+** | 50 | 17.50 | 5.84 | 0.33 | 45.62 | 17.50 | +28.12 | 86.37% |

## Representative Failure Cases
A failure mode emerges clearly in the 16+ bucket. For example, a target sequence of 17 tokens requires a theoretical minimum of 17-18 timesteps. The model's actual temporal width $T$ for the word image is ~45. The network has plenty of room to decode the word (Margin > +28). However, the network *only* outputs ~5 tokens, spending the remaining 86% of the timesteps outputting high-confidence `[BLANK]` predictions.

## Interpretation
The evidence categorically disproves the theory that the model physically runs out of CTC timesteps for long words. 
Across all buckets, $T$ exceeds $\text{Required\_}T$ by roughly ~30 timesteps.
Instead, we observe severe **Temporal Saturation / Feature Collapse**. Regardless of whether the word is 4 tokens long or 18 tokens long, the network restricts its non-blank outputs to roughly 5-6 positions and defaults to `[BLANK]` for ~86% of the sequence.

## Conclusion

**A. Is temporal capacity insufficient for long words?**
No. The actual CTC timesteps ($T$) consistently exceed the theoretical minimum required timesteps by a wide margin (usually by +25 to +35 timesteps) across all word lengths.

**B. Are long-word failures caused by $T < \text{Required\_}T$, or does the model still have enough theoretical CTC timesteps but fail to use them?**
The model has ample theoretical timesteps but completely fails to use them.

**C. Is there evidence of blank-dominant/saturated temporal representations?**
Yes. The blank probability remains saturated at ~85-87% across *all* word lengths. The model is collapsing spatial features too aggressively, forcing multiple characters into single temporal receptive fields, causing confusion and defaulting the rest of the sequence to blanks.

**D. Does the evidence justify investigating temporal-resolution/feature-width changes in a FUTURE controlled training experiment?**
Yes. Since the capacity exists but the features are collapsing, the current $W/4$ downsampling via max-pooling might be destroying the spatial separability of characters in cursive Tamil script. A future controlled experiment that reduces horizontal pooling (e.g. $W/2$ or $W/1$) to preserve finer spatial feature separability is heavily justified.

## Limitations
This diagnostic did not measure internal CNN feature variance or gradient flow, only the final CTC logits. It assumes the CTC alignment correctly reflects the CNN spatial mapping without significant receptive field distortion.
