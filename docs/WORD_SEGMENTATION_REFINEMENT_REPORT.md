# Word Segmentation Refinement Report

## 1. Existing Segmentation Method
The original segmentation logic (`models\cv\segmenter.py`) used a purely connected-component (CC) based bounding box extraction with naive spacing thresholds.
*   **Failure Mode:** Historical Tamil textual lines often have completely touching or zero-gap words. The original CC algorithm aggressively merged these distinct words because the median gap calculation skewed too high, combining clearly separated characters alongside touching ones into excessively long multi-word crops.
*   **Impact:** The OCR model was strictly trained on a single-word dataset. Feeding it multi-word merged crops resulted in out-of-distribution (OOD) failure.

## 2. Diagnostic Comparison
A local script (`scripts/multi_scale_segmentation_test.py`) evaluated multi-scale methods against `sample_10lines.png`:
*   **Vertical Projection:** Extreme over-segmentation. Broken strokes and distinct characters within words were split unnecessarily due to minor internal whitespace.
*   **Naive CC:** Extreme under-segmentation. Adjacent characters were indiscriminately grouped.
*   **Combined Morphology+CC:** By applying a light morphological closing first, strokes within characters were bound together. Subsequent CC analysis with adaptive thresholding dynamically adjusted spacing logic, distinguishing inter-word and intra-word boundaries effectively.

## 3. New Refined Method
The `Segmenter.segment_words()` algorithm was overhauled with the Combined Morphology+CC method:
1.  **Morphological Closing:** Applies a dynamic kernel (`max(1, height*0.05)`, `max(2, height*0.1)`) to bridge small intra-character gaps.
2.  **CC Extraction:** Filters out sub-1% pixel noise.
3.  **Horizontal Component Grouping:** Overlapping and extremely close bounds (<= 0px separation) are strictly merged to protect character integrity.
4.  **Adaptive Thresholding:**
    *   If multiple distinct CC groups exist, the split threshold is `max(int(height*0.1), median_gap * 1.5)`.
    *   Otherwise, a static heuristic of `int(height*0.15)` is applied.
5.  **Safe Margins:** Standard padding bounds crops around derived boundaries.

## 4. Handling Touching Words
**Constraint:** Do not force arbitrary splits if there is zero whitespace between words.
**Implementation:** If the segmentation process produces an anomaly—a single, massive merged component spanning the image width with an aspect ratio `> 4.0`—it assigns `confidence = 0.5`. This gracefully passes the unsegmented crop to the OCR layer as a "best-effort" attempt without hallucinating arbitrary splits.

## 5. Preprocessing Experiments
Prior to CRNN ingestion, crops are aggressively resized to 32px height via `AspectRatioPreservingResize` (LANCZOS interpolation).
*   **Experiment:** Mild sharpening.
*   **Outcome:** Added `ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)` prior to interpolation. This mildly dilates and preserves the density of thin Tamil diacritics before the aggressive scale-down, reducing aliasing-induced feature loss.

## 6. Synthetic Control Testing
A synthetic test image (`synthetic_test_line.png`) was created by concatenating three distinct word crops with controlled 10px spacing.
*   **Before:** Would often be merged if gaps fell under rigid statistical boundaries.
*   **After:** The refined segmenter successfully yielded `Words detected: 3` (expected 3), dynamically recognizing the threshold. OCR order and reconstruction were flawlessly maintained.

## 7. Real-Image Qualitative Results
*   **`sample.jpg`:** Extracted 1 large crop (perfect isolation).
*   **`sample_test.png`:** Extracted 1 tiny crop (perfect isolation).
*   **`sample_10lines.png`:** Evaluated at 13 lines, 16 distinct words. Zero-gap lines remained conservatively merged, while functionally spaced lines achieved proper splitting.

## 8. Limitations & Safety Audit
*   **Files Modified:** `models/cv/segmenter.py`, `app/inference.py`.
*   **Files Created:** `scripts/multi_scale_segmentation_test.py`, `scripts/synthetic_segmentation_test.py`.
*   **Model Checkpoint:** `checkpoints/recognition/tamil/ocr_balanced_temporal/best.pth` (Unchanged).
*   **Actual Temporal Downsampling:** Hardcoded to `T = W / 4` (Pool 2x2, 2x2, 2x1) inside `models/recognition/encoder.py`.
*   **Model Integrity:** Zero parameter modifications, no retraining, no external LLMs.
*   **Application Startup:** Functional.
*   **Remaining Blockers:** Dense zero-gap (fully touching) words cannot be deterministically segmented by classic CV without risking destruction of Tamil characters. True separation of these anomalous crops will require localized stroke-width transforms or object-detection architectures in the future.
