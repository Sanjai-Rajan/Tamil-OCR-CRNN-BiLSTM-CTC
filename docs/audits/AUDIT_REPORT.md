# Phase 1: Full Code Audit Report

## 1. Current Architecture
The OCR model utilizes a Convolutional Recurrent Neural Network (CRNN) with Connectionist Temporal Classification (CTC).
- **Encoder:** A deep 6-layer CNN (64 -> 128 -> 256 -> 256 -> 512 -> 512 channels) with BatchNormalization and ReLU activations. It applies $2\times2$ MaxPooling at the first two layers, and $2\times1$ MaxPooling at the 4th layer. This results in a $4\times$ horizontal downsampling and an $8\times$ vertical downsampling.
- **Sequence Model:** A 2-layer Bidirectional LSTM with 256 hidden units.
- **Prediction Head:** A fully connected linear layer mapping the 512-dimensional output of the BiLSTM to 78 classes (77 characters + 1 CTC blank).

## 2. Current Data Flow
1. **Preprocessing:** Grayscale conversion (`L` mode), Aspect Ratio Preserving Resize to a fixed height of 32 pixels. The width is allowed to vary up to a maximum. Pixel values are normalized to $[-0.5, 0.5]$.
2. **Batch Collation:** Images in a batch are padded with white space (`1.0` value) on the right to match the maximum width `W` present in that specific batch.
3. **Model Forward Pass:** The padded tensor `(B, 1, 32, W)` passes through the Encoder, outputting `(B, 512, 4, W/4)`. It is reshaped to `(B, W/4, 2048)` and passed to the BiLSTM, outputting `(B, W/4, 512)`, and finally to the Prediction Head outputting `(B, W/4, 78)`.
4. **CTC Calculation:** The network utilizes PyTorch's native `CTCLoss(blank=0, zero_infinity=True)`. The `input_lengths` are accurately computed as `actual_width // 4` to prevent the loss from penalizing the padded regions.
5. **Decoding:** Simple greedy argmax decoding (`torch.argmax(dim=2)`), followed by standard CTC blank and duplicate collapsing.

## 3. Things That Are Already Correct
- **CTC Implementation:** Blank index correctly maps to `0`. `input_lengths` correctly ignore batch padding. 
- **Decoding & Loss:** The argmax decoding is standard and correctly collapses duplicates. `zero_infinity=True` correctly protects against infinite loss when targets are longer than input sequences.
- **CNN Spatial Downsampling:** A $4\times$ horizontal downsampling is completely appropriate. The average ground-truth sequence is ~10 characters, and the average CNN output is ~40 timesteps. This provides a very healthy ratio of ~4 timesteps per character.

## 4. Confirmed Implementation Issues & Bottlenecks
1. **Severe Padding Inefficiency:** The data loader batches images randomly without considering their widths. This forces narrower images to be heavily zero-padded to match the widest image in the batch. Our diagnostics show an `avg_actual_to_padded_ratio` of `0.55`, meaning **45% of the tensor processed by the A100 is dead padding**.
2. **Extreme Class Imbalance:** Certain characters appear nearly 10,000 times in validation, while others appear fewer than 5 times. The model completely fails to predict 4 minority characters.

## 5. Suspected Causes of Blank Dominance
The CTC "blank dominance" (measured at ~75-86%) is **NOT a bug**. 
It is mathematically enforced by the architecture and data flow. Because the model outputs ~40 timesteps for an average sequence of 10 characters, it *must* emit blanks for the remaining ~30 timesteps to separate and align the sequence. 
Our diagnostics proved that the model is highly confident when outputting blanks for correct predictions (avg prob 76.7%), demonstrating that it has learned a highly accurate, spiky probability distribution. 
However, the massive amount of zero-padding (issue #1) forces the model to constantly predict blanks for trailing empty space, unnecessarily reinforcing the blank bias.

## 6. Recommended Experiments Ranked by Expected Impact
1. **EXPERIMENT B (Input-Width Handling & Bucketing):** Implement dynamic aspect-ratio batch bucketing. Grouping images of similar widths will eliminate the 45% padding waste, effectively doubling the A100's throughput. This is the highest impact, lowest risk change.
2. **EXPERIMENT D (CTC Dominance Mitigation - Focal Loss):** Implement a Focal CTC loss or class weighting scheme to combat the extreme class imbalance and recover the missing minority characters.
3. **EXPERIMENT C (Reduce CNN Spatial Downsampling):** Test reducing the horizontal downsampling from $4\times$ to $2\times$. This will double the number of timesteps available to the CTC decoder, potentially improving alignment for tightly packed characters.
4. **EXPERIMENT E (Character-Focused Training):** Investigate `TamilChar.csv` to see if pretraining the CNN on individual characters yields better feature extraction before sequence training.
