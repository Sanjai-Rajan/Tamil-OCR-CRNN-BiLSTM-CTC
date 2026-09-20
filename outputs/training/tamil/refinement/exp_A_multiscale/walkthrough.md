# Multi-Scale Augmentation (Exp A) Results

We implemented and evaluated **Experiment A (Multi-Scale Augmentation)** to test if injecting randomly scaled and padded text into the training pipeline would force the CRNN to learn scale-invariant features.

The results show a catastrophic failure. The model suffered severe mode collapse, completely losing the ability to read even full-sized text.

## 1. Validation Metrics

| Metric | Baseline | Exp A | Absolute Change | Relative Change |
| :--- | :--- | :--- | :--- | :--- |
| **CER** | 0.3583 | 0.9467 | +0.5884 | +164.21% |
| **WER** | 0.8140 | 1.0000 | +0.1860 | +22.85% |
| **Character Accuracy** | 0.6417 | 0.0533 | -0.5884 | -91.69% |
| **Word Accuracy** | 0.1860 | 0.0000 | -0.1860 | -100.00% |
| **Validation Loss** | ~3.40 | 3.4187 | - | - |

> [!WARNING]
> While the CTC Loss reached a mathematically "acceptable" value (3.41), the CER skyrocketed to 94.6%. This indicates **CTC Mode Collapse**, where the model learned to predict blanks almost everywhere to minimize loss on the massive amounts of white padding we introduced.

## 2. Scale Robustness Test

We tested the models on the same training image scaled into a progressively larger white canvas.

| Scale | Baseline Prediction | Baseline Correct? | Exp A Prediction | Exp A Correct? |
| :--- | :--- | :--- | :--- | :--- |
| **100%** | இவுங்க | YES | க | NO |
| **75%** | இறைவ்ப் | NO | க | NO |
| **50%** | நி | NO | க | NO |
| **37.5%** | உஸ்ப் | NO | க | NO |
| **25%** | ட் | NO | க | NO |

Exp A completely collapsed to predicting a single character ("க") regardless of scale.

## 3. Sample Test (`sample_test.png`)

**Baseline Prediction:** `காலன்`
**Exp A Prediction:** `க`

## 4. Training Analysis

- **Total training time:** ~7185s (approx. 2 hours)
- **GPU used:** cuda
- **Best epoch:** 2
- **Final training loss:** 3.534
- **Final validation loss:** 3.418
- **Best validation metric (CER):** 0.9467
- **Overfitting occurred:** NO
- **Checkpoint path:** `checkpoints/recognition/tamil/exp_A_multiscale/best.pth`

## 5. Decision: REGRESSION

**Classification: D. REGRESSION**

**Explanation:** 
Experiment A is a massive regression. By randomly shrinking the text and placing it onto the original-sized white canvas, we introduced enormous amounts of horizontal and vertical whitespace. 
Because a CRNN scans the image horizontally, it spent 75%+ of its receptive field scanning pure white pixels. It naturally converged on a degenerate solution (Mode Collapse): predicting blanks for the white space and outputting a single dominant character ("க") for any dark pixels it encountered. 

We must discard this model. The scale-robustness issue cannot be solved by naively adding massive padding to the training images, as this breaks the CRNN's horizontal sequence alignment.
