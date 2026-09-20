# OCR Balanced Temporal — Real Image Inference Report

## 1. Overview
This report evaluates the inference pipeline behavior using the newly trained `ocr_balanced_temporal/best.pth` checkpoint on a set of real-world images. The test focuses on verifying the checkpoint loading, image preprocessing, line segmentation (for multi-line text), and OCR model decoding using the modified `less_downsample=True` (W/8) architectural configuration.

## 2. Checkpoint Details
- **Path:** `checkpoints/recognition/tamil/ocr_balanced_temporal/best.pth`
- **Architecture Config:** CRNN with `less_downsample=True` (Temporal dimension = W/8).
- **Status:** Checkpoint loaded successfully with PyTorch `weights_only=False`. (There was an initial issue with numpy safe globals that required `weights_only=False` to parse the saved optimizer states correctly).

## 3. Real Image Test Results

### 3.1 `sample.jpg`
- **Detected Lines:** 1
- **Preprocessing:** Resized vertically to 32px preserving aspect ratio.
- **Prediction:** `பொடு` (Note: highly degraded image, prediction heavily character-biased towards single short words due to the short prediction length collapse seen in balanced_temporal validation).
- **Status:** Completed.

### 3.2 `sample_test.png`
- **Detected Lines:** 1
- **Prediction:** `ஸாம்`
- **Status:** Completed.

### 3.3 `sample_10lines.png` (Multi-line document)
- **Segmentation Strategy:** Simple horizontal projection profile using Otsu thresholding.
- **Lines Detected:** 10 lines (Successfully identified every line).
- **Line Ordering:** Top-to-bottom reading order was perfectly preserved.
- **Reconstructed Text:**
```
LINE 1: வாய்
LINE 2: டீ்
LINE 3: ந்ர்்
LINE 4: அனு்
LINE 5: கடத்
LINE 6: வ்
LINE 7: பழங்க்
LINE 8: கீ
LINE 9: ட்
LINE 10: ங்
```

## 4. Pipeline Verification
- **Checkpoint Loading:** **PASS** (Successfully loaded).
- **Preprocessing:** **PASS** (Resized to 32px height and normalized appropriately).
- **Segmentation:** **PASS** (10 lines cleanly separated from the 10-line source document).
- **Line Ordering:** **PASS** (Top-to-bottom ordering preserved).
- **OCR Output:** **PASS/FLAGGED** (The inference successfully generated output; however, the actual recognized text severely collapsed into extremely short snippets and mode-collapsed characters, matching the `CER: 0.50` character instability documented in training).
- **Reconstructed Text:** **PASS** (Lines concatenated properly).

## 5. Conclusion
The inference pipeline technically functions flawlessly with the `ocr_balanced_temporal` architecture (provided `less_downsample=True` is set in the model initialization). However, the actual qualitative OCR text produced by the model is highly degraded and dominated by mode-collapse (frequent repetition of `்`, extremely short 1-3 character tokens, and heavy hallucination on longer lines), indicating that the `ocr_balanced_temporal` model does not generalize well to complex real-world multi-line documents compared to earlier stabilizing experiments.
