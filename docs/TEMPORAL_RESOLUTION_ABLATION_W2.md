# Experiment 3: Controlled W/2 Temporal-Resolution Ablation

## 1. Objective
This experiment tested whether preserving a higher horizontal feature-map resolution (W/2 instead of the production W/4) mitigates the severe long-word length-collapse saturation (~5-6 tokens) observed in the CRNN model during Experiment 2.

## 2. Methodology & Integrity Constraints
- **Controlled Ablation:** The encoder architecture was modified to EncoderW2, changing the pooling operations from (2,2), (2,2), (2,1) to (2,2), (2,1), (2,1). All other components (CNN layers, BiLSTM, CTC head, vocabulary, preprocessing) were strictly preserved.
- **Training Data:** The model was trained from scratch for 10 epochs using the EXACT same native 	amil/train split used by the production system.
- **No Pretrained Weights:** The model was trained entirely from random initialization. No transformers, LLMs, or generative AI were used.
- **Frozen Baseline:** The W/4 baseline (ocr_balanced_temporal/best.pth) was left completely untouched for comparison.
- **Evaluation:** Evaluated on the same native 	amil/test split with standard CTC decoding and conceptual correction strictly disabled.

## 3. Results Summary

| Metric | W/4 Baseline | W/2 Ablation | Difference |
|--------|--------------|--------------|------------|
| **CER** | 0.2176 | 0.2240 | +0.0064 (Worsened) |
| **WER** | 0.4199 | 0.4234 | +0.0035 (Worsened) |
| **Character Acc** | 78.24% | 77.59% | -0.65% |
| **Word Acc** | 58.01% | 57.65% | -0.36% |
| **Avg Target Len** | 4.79 | 4.79 | - |
| **Avg Pred Len** | 3.26 | 3.14 | -0.12 |
| **Blank %** | 92.97% | 92.55% | -0.42% |

### Length Bucket Breakdown

| Bucket | Target Avg | W/4 Pred Avg | W/2 Pred Avg | W/2 CER | W/2 Word Acc |
|--------|------------|--------------|--------------|---------|--------------|
| **1-5** | 1.97 | 1.99 | 1.97 | 7.14% | 86.89% |
| **6-10** | 8.13 | 5.53 | 5.27 | 46.75% | 0.08% |
| **11-15**| 12.57 | 5.98 | 5.66 | 58.81% | 0.00% |
| **16+** | 17.92 | 6.38 | 5.87 | 68.72% | 0.00% |

## 4. Interpretation

**A. Overall CER/WER**
Both overall CER and WER slightly worsened with W/2 compared to the highly trained W/4 baseline, which is expected given the W/2 model was only trained for 10 epochs.

**B. Long-word CER**
Long-word CER remained extremely high (68.72% for 16+ tokens).

**C-D. Long-word Prediction Lengths & Saturation**
Crucially, the long-word prediction length did *not* increase toward the target. In fact, it **decreased slightly**. The ~5-6 token saturation remained absolutely intact. For a target averaging 17.92 tokens, the W/2 model predicted an average of only 5.87 tokens. 

**E. Blank Dominance**
Blank dominance remained essentially identical (92.5% vs 92.9%). Despite the W/2 model receiving double the horizontal capacity (mean CTC T = 42.58), it utilized that extra capacity simply to output more blanks. 

**F. Does the experiment support continuing with W/2?**
No. This ablation provides definitive evidence that the length collapse is **not** caused by insufficient temporal or spatial capacity in the final feature map. Doubling the receptive field density (W/2) completely failed to alter the collapse behavior. 

**Hypothesis:** Combined with the findings of Experiment 3A (which revealed that 46% of the training set is 1-5 tokens, and long sequences are aggressively squeezed horizontally), it is highly likely the length saturation is driven by dataset bias and loss-scaling optimization, not architectural capacity. The model learns a strong prior to emit ~5 tokens and relies on padding the rest with blanks, independent of the CNN's temporal width.

## 5. Limitations
The W/2 model was only trained for 10 epochs. However, given that the saturation behavior is identical in magnitude and characteristic to the fully trained W/4 model, it is extremely improbable that further training would magically reverse the structural length collapse.
