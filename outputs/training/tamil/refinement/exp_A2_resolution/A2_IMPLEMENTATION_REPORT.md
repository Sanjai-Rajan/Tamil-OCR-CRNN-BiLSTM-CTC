# A2 Implementation Report: Controlled Resolution Degradation

## A. Augmentation Algorithm
The `ResolutionDegradation` augmentation simulates low-resolution inputs without modifying the bounding box or aspect ratio of the text. It randomly downsamples the tightly-cropped training image to a fraction of its original size and immediately upsamples it back to the original dimensions. To prevent the model from overfitting to the exact `0px` boundary, a very small random padding (0-10% of the image height) is applied *before* degradation, so the padding itself is degraded and naturally integrates with the image.

## B. Probability Distribution
Configured precisely to the requested distribution:
- 30%: Original (No degradation)
- 20%: Mild (~75% scale)
- 20%: Moderate (~50% scale)
- 15%: Strong (~37.5% scale)
- 10%: Very strong (~25% scale)
- 5%: Extreme (~20% scale)

## C. Interpolation Method
- **Downsampling**: `Image.Resampling.BILINEAR`. Chosen to simulate natural optical/sensor loss of high-frequency information.
- **Upsampling**: `Image.Resampling.LANCZOS`. Chosen to simulate the high-quality `AspectRatioPreservingResize` interpolation used in the production inference pipeline.

## D. Preview Findings
Generated previews for 20 training samples across all 6 scales.
- **Preservation**: The text remains tightly framed (or with a maximum 10% natural-looking margin). Aspect ratio is perfectly preserved. 
- **Diacritics**: Pulli and upper/lower marks (like in "க்" or "ஹ") are preserved but appropriately blurred.
- **No Clipping/Explosion**: No extreme padding is introduced, avoiding the CTC mode-collapse seen in Experiment A.

## E. Baseline Diagnostic Predictions
Before training, the baseline model was evaluated offline against the degraded variants of 3 training samples. 
Interestingly, **the baseline model handles pure blur quite well if the framing remains tight**:
- Sample 0 (`தொலர்ததினம்`): Maintained the exact same prediction down to 20% scale.
- Sample 1 (`பேரினள`): Shifted slightly to `பேரியள` at 75% scale and stayed there down to 20% scale.
- Sample 2 (`இவுங்க`): Maintained 100% accurate prediction down to 20% scale.

## F. Smoke-Test Results
The `verify_one_batch` routine successfully loaded an augmented batch of 32 images. 
- **Loss**: Initial CTC loss was stable (18.27). 
- **Tensors**: `Images shape: torch.Size([32, 1, 32, 231])`. Batching works properly.
- **Labels**: Decoded seamlessly without NaN errors.
- **GPU**: CUDA memory allocations operated smoothly.

## G. Sample_Test Simulation & Expected Benefits
In `sample_test.png`, the detected text is roughly 100x8. When upscaled 4x to 32px height by the inference pipeline, it becomes heavily blurred. 
The **Very Strong (25%)** and **Extreme (20%)** degradation levels closely simulate the loss of edge crispness seen in `sample_test.png`. By training on this mix, the CRNN's CNN backbone is expected to learn to rely on global structural topology rather than highly-localized, crisp edges.

## H. Risks & Observations
While `ResolutionDegradation` accurately simulates low-resolution blur, **it does not perfectly simulate relative stroke-thickening**. 
- In `sample_test.png`, a 1px stroke in the 8px image becomes a 4px stroke in the 32px tensor (12.5% of the total height).
- In our training data (where text is natively 300px tall with ~10px strokes), downsampling to 75px and upsampling back to 300px blurs the stroke, but it remains ~10px thick geometrically (still 3.3% of the total height). 
Therefore, if `sample_test.png` is failing primarily due to physical stroke *thickness* rather than just *blurriness*, A2 alone might only partially solve the problem. If A2 fails to completely solve `sample_test.png`, the next logical step would be adding morphological dilation (thickening the strokes) before degradation.

## I. Recommended Full-Training Configuration
We are ready to launch the full run:
```bash
python scripts/train_refinement.py --use-res-degradation --experiment exp_A2_resolution --epochs 40
```
This will train the baseline CRNN on the existing dataset with the A2 augmentation active.
