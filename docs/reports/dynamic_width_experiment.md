# Dynamic Width Diagnostic Report

## Objective
The previous experiments showed severe CTC mode-collapse, resulting in the model overwhelmingly predicting blank tokens. A critical hypothesis was that the fixed-size resizing (forcing all variable-length Tamil words into a 32x128 pixel box) aggressively destroyed the image aspect ratio. After CNN downsampling by a factor of 4, the temporal dimension $T$ was fixed at 32. Since CTC requires $T \ge 2 \times \text{target\_length} + 1$, many longer Tamil words mathematically violated this constraint, forcing the model into an unrecoverable blank-prediction collapse.

This diagnostic tested replacing the fixed 128px width with dynamic, aspect-ratio preserving widths and batch-level padding.

## 1. Visual Verification
Old preprocessing aggressively squashed text horizontally, destroying morphological features of the Tamil script. The new dynamic resizing algorithm successfully scales all images to a height of 32px while preserving their natural aspect ratio, padding shorter images with white (1.0) up to the maximum width in the batch.

*(Reference: `outputs/diagnostics/preprocessing_comparison.png`)*

## 2. CTC Safety Verification
We executed a diagnostic across the 500-sample subset to calculate the new available temporal dimension ($T = \lfloor \frac{\text{width}}{4} \rfloor$) and compared it against the minimum CTC required length.

**Results:**
- **Samples Satisfying CTC Constraint:** 500 / 500
- **Samples Violating CTC Constraint:** 0
- **Maximum Target Length Observed:** 30 characters
- **Minimum Available $T$:** 11 (for very short words)

The new dynamic width preprocessing completely resolves the mathematical CTC sequence length violations.

## 3. Tiny-Dataset Overfit Diagnostic (30 Epochs)
With the CTC violations resolved, we re-ran the 500-sample overfit diagnostic to determine if the architecture could now memorize the dataset.

**Configuration:**
- Batch Size: 32
- Learning Rate: 0.0003
- Optimizer: Adam
- Dataset: 500 samples (same for train/eval)

**Results:**
- **Epoch 01:** Train Loss: 7.7686 | CER: 1.0000 | Blanks: 100.0%
- **Epoch 10:** Train Loss: 3.4100 | CER: 1.0000 | Blanks: 100.0%
- **Epoch 20:** Train Loss: 3.3295 | CER: 1.0000 | Blanks: 100.0%
- **Epoch 30:** Train Loss: 3.2940 | CER: 0.9455 | Blanks: 98.6% | Unique Seqs: 1

## Conclusion and Final Recommendation
**Conclusion:**
While the dynamic width preprocessing successfully fixed the critical CTC length violations and preserved the morphological integrity of the images, the CRNN + BiLSTM architecture *still completely fails to overfit a 500-sample dataset*. The model continues to collapse into predicting nearly 100% blank tokens and producing only a single unique sequence. 

**Diagnosis:**
Since the data pipeline, vocabulary, CTC blank index, and temporal dimension requirements are now proven correct, the root cause of the collapse lies entirely within the neural network architecture or its optimization dynamics. The model is failing to learn the image-to-text mapping even under ideal overfitting conditions.

**Final Recommendation:**
**DO NOT resume full dataset training.**
The next step must focus exclusively on the **CRNN architecture and initialization**. We need to perform an architectural diagnostic focusing on:
1. **CNN Receptive Field:** Does the current CNN backbone extract sufficiently distinct features before passing them to the RNN?
2. **BiLSTM Capacity/Initialization:** Are the recurrent layers suffering from vanishing/exploding gradients early in training?
3. **Weight Initialization:** The model may require explicit initialization (e.g., Xavier/Kaiming) rather than PyTorch defaults to break initial symmetry.
4. **Learning Rate/Optimizer:** The optimizer might be stuck in a local minimum instantly. 

We should isolate a single batch of 4 images and modify the architecture/initialization until the model can achieve 0.0 CER on that single batch.
