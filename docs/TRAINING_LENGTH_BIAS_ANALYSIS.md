# TRAINING LENGTH BIAS ANALYSIS

## 1. Objective
Perform a diagnostic study of the Tamil OCR training and evaluation pipelines to determine if the long-sequence length collapse observed in inference (~5-6 token saturation) is associated with the training target-length distribution.

## 2. Training Length Distribution
*Total Training Samples*: **126,419 valid samples** (out of total dataset space)
Target length distribution:
- **1–5**: 58,322 (46.13%)
- **6–10**: 39,723 (31.42%)
- **11–15**: 22,649 (17.92%)
- **16+**: 5,725 (4.53%)

Additional extreme long-word counts:
- **>= 20**: 1,271 samples
- **>= 25**: 147 samples
- **Maximum target length**: 32 tokens

## 3. Train vs Test Distribution Comparison
*Test Samples: 43,110*

| Bucket | Training Samples | Training % | Test Samples | Test % |
|--------|------------------|------------|--------------|--------|
| **1–5** | 58,322 | 46.13% | 33,674 | 78.11% |
| **6–10** | 39,723 | 31.42% | 7,358 | 17.07% |
| **11–15**| 22,649 | 17.92% | 1,745 | 4.05% |
| **16+** | 5,725 | 4.53% | 333 | 0.77% |

*Note: The native TEST split is even more heavily skewed towards extremely short sequences (78% vs 46%), but the long-tail behavior is consistently sparse across both splits.*

## 4. Length-Wise Recognition Behaviour & Prediction Collapse
Using the frozen W/4 production baseline (ocr_balanced_temporal/best.pth):

| Bucket | Target Avg | W/4 Pred Avg | Pred/Target Ratio | CER | Word Acc |
|--------|------------|--------------|-------------------|-----|----------|
| **1–5** | 1.97 | 1.99 | 1.010 | 6.96% | 87.30% |
| **6–10** | 8.13 | 5.53 | 0.680 | 45.34% | 0.52% |
| **11–15**| 12.57 | 5.98 | 0.476 | 57.20% | 0.00% |
| **16+** | 17.92 | 6.38 | 0.356 | 66.63% | 0.00% |

**Prediction-Length Collapse Observation:** 
The shortening behavior becomes progressively stronger for longer targets. While 1-5 token words predict nearly exactly the right length (Ratio 1.01), a 17-token word is crushed into just 6.38 tokens on average (Ratio 0.356). The model asymptotically flatlines at ~6 output tokens regardless of input sequence length.

## 5. Training-Exposure Analysis (Long-Sequence Subset)
The model's exposure to sequences >10 tokens is mathematically limited:
- **11–15 token samples**: 17.92% of training set
- **16+ token samples**: 4.53% of training set

The model sees ~10x more extremely short (1-5) sequences than it does 16+ sequences. It overwhelmingly learns a prior that most valid words finish quickly, rewarding it for predicting blanks early.

## 6. Diagnostic Conclusion
The extreme skew of the training target distribution (where nearly 50% of the data consists of sequences under 5 tokens, and long sequences >15 tokens represent <5% of the data) is highly **consistent with** a short-sequence training bias. Combined with the previously confirmed failure of the W/2 temporal ablation, this data **supports investigating** a controlled length-aware batch-sampling experiment.

*Disclaimer: This distribution imbalance does not definitively establish causality, as image resizing constraints (width plateaus at ~1500px) also simultaneously crush horizontal feature density for long words. However, the skewed prior strongly correlates with the observed CTC output collapse.*
