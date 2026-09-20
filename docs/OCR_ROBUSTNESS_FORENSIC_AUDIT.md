# Controlled Robustness Forensic Audit

## 1. Baseline
The currently verified results for `checkpoints\recognition\tamil\ocr_balanced_temporal\best.pth` are:
- **Samples:** 43,110
- **CER:** 21.76%
- **WER:** 41.99%
- **Character Accuracy:** 78.24%
- **Word Accuracy:** 58.01%

## 2. Sample Test Images
A deterministic random sample across target length buckets yields the following characteristics (from `robustness_samples.json`):

### Bucket: 1-5 tokens
- **Target Len:** 4-5
- **Aspect Ratios:** 3.6x to 11.8x
- **Original Widths:** ~1,200 to ~3,000 px (Heights ~250-350 px)
- **Model Input Widths:** 116 to 379 px
- **Temporal Lengths:** 29 to 94 frames
- **Truncated:** False

### Bucket: 16+ tokens
- **Target Len:** 16-19
- **Aspect Ratios:** 3.9x to 7.8x
- **Original Widths:** ~1,200 to ~1,900 px (Heights ~240-380 px)
- **Model Input Widths:** 111 to 251 px
- **Temporal Lengths:** 28 to 62 frames
- **Truncated:** False

## 3. Preprocessing Forensics
The preprocessing applies an `AspectRatioPreservingResize` to a fixed height of `32px`.
- **Horizontal Crushing:** No direct algorithmic crushing occurs, as the aspect ratio is mathematically preserved.
- **Stroke Loss:** Original images are extremely high resolution (e.g., 2,999x253). Scaling a 253px high image down to 32px requires heavy downsampling (a factor of ~8x). Thin Tamil strokes and diacritics are highly susceptible to loss or blurring during this aggressive downsampling.
- **Padding:** Present during batching, but does not distort the word structure itself.

## 4. Segmentation Forensics
Diagnostics from `sample.jpg`, `sample_test.png`, and `sample_10lines.png`:
- **sample.jpg**: 1 word correctly segmented (512x128).
- **sample_test.png**: 1 word correctly segmented (110x20).
- **sample_10lines.png**: Exhibits severe under-segmentation (merging). For instance, lines 1 and 3 yielded only a single word candidate despite containing multiple distinct words visually.
- **Cause:** Historical Tamil text often features near-zero spacing or physical touching between characters of adjacent words. The segmenter merges these into single, excessively wide image crops.

## 5. Model Input Forensics
The transformation pipeline dictates temporal length as follows:
1. `new_w = max(4, round(orig_w * 32 / orig_h))`
2. CRNN applies two pooling layers (`less_downsample=True`), yielding `temporal_length = new_w // 4`.

**CTC Feasibility:**
In all tested samples—including the extreme 16+ bucket—the `temporal_length` significantly exceeds the `min_ctc_len`.
*Example:* A 19-token target yielded an original width of 1923px and height of 245px.
`new_w = round(1923 * 32 / 245) = 251`.
`temporal_length = 251 // 4 = 62`.
Since 62 > 19, this is mathematically feasible under CTC constraints.
**Conclusion:** The model is not failing due to CTC mathematical impossibility. It is failing due to poor sequence resolution or receptive field mapping for highly compressed, multi-word (merged) crops that it was not trained on.

## 6. Correction Forensics
Analysis of `models\digitalization\conceptual_corrector.py`:
- **Vocabulary Size:** 522,495 unique classical Tamil words.
- **Minimum Frequency / Correction Threshold:** 10 occurrences required to validate a candidate.
- **Edit Distance:** Exactly 1 (no unbounded fuzzy matching).
- **Fallback Behavior:** Gracefully returns the original raw OCR prediction if no candidate meets the threshold.

## 7. Final Diagnosis (Top Bottlenecks)
Ranked strictly by forensic evidence:

### 1. Under-Segmentation (Word Merging)
- **Evidence:** `sample_10lines.png` produces single-word crops for lines with multiple words.
- **Pipeline Stage:** `models\cv\segmenter.py`
- **Severity:** CRITICAL. The CRNN was trained on single-word crops. Feeding it multi-word merged crops causes catastrophic out-of-distribution failure.
- **Can it be fixed without retraining?** YES. Requires a more robust deterministic word segmentation approach (e.g., adaptive spacing thresholds or contour analysis).

### 2. Excessive Downsampling (Loss of Detail)
- **Evidence:** Original images scale from heights of ~300px down to 32px (almost a 10x reduction).
- **Pipeline Stage:** Preprocessing (`AspectRatioPreservingResize(32)`).
- **Severity:** HIGH. Thin Tamil diacritics (which differentiate characters) are easily destroyed when interpolating down to 32px.
- **Can it be fixed without retraining?** YES/PARTIALLY. Can be mitigated by careful sharpening filters, morphological operations, or tweaking interpolation methods, though fundamental receptive field limits still apply.

### 3. Receptive Field Mismatch for Merged Words
- **Evidence:** Although CTC is mathematically feasible (e.g., 62 frames for 19 characters), packing 19 characters into 62 frames gives each character only ~3 frames of representation. The network may struggle to resolve features at this density if it primarily learned on shorter words with wider frame-per-character mappings.
- **Pipeline Stage:** CRNN Inference.
- **Severity:** HIGH.
- **Can it be fixed without retraining?** YES. Fixing the segmentation (Bottleneck #1) automatically resolves this by ensuring the CRNN only receives expected single-word distributions.
