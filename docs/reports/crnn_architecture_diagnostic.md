# CRNN Architecture Diagnostic Report (Single-Batch)

## Objective
To determine why the CRNN cannot overfit a tiny dataset despite fixing the CTC temporal dimensions. We isolated exactly 4 samples and ran a deep architectural audit on the gradients, initializations, and feature diversity across the CNN, BiLSTM, and Linear Head.

## Diagnostic Results

### 1. Variable Width / Masking Audit
- **Padded Batch Dimensions:** `[4, 1, 32, 246]`
- **Actual Image Widths:** `[246, 141, 164, 186]`
- **CNN Output Temporal Dimension (T):** `61`
- **CTC Input Lengths:** `[61, 35, 41, 46]`
**Finding:** The variable width logic and CTC sequence masking are mathematically perfect. The CTC head is correctly ignoring the padded white space. Masking/padding is **NOT** the issue.

### 2. Feature Diversity Audit (The Smoking Gun)
We extracted the representations from the CNN and the BiLSTM for the 4 visually distinct images before any training occurred:
- **CNN Cosine Similarity Mean:** `0.9953` (99.5% identical)
- **LSTM Cosine Similarity Mean:** `0.9999` (99.99% identical)

**Finding:** The CNN encoder is mapping completely different manuscript images into **virtually identical feature vectors**. The BiLSTM then takes these indistinguishable features and makes them perfectly identical. 
Because the network produces the exact same features for different images, the CTC loss is mathematically forced to predict the exact same sequence for everything. Since the target sequences are different, the only safe local minimum is to predict 100% BLANKS to minimize error across conflicting targets.

### 3. Gradient Flow Audit (Epoch 200)
- **CNN (`encoder.features.0`):** Gradient Norm = `1.77e-07` (Absolute Zero)
- **BiLSTM (`sequence.lstm`):** Gradient Norm = `1.35e-06` (Effectively Zero)
- **CTC Head (`head.linear`):** Gradient Norm = `0.27` (Healthy)

**Finding:** There is massive **Vanishing Gradients** in the architecture. The error from the CTC head cannot propagate backward through the LSTM into the CNN. The CNN weights are essentially frozen at their random initialization, and since that initialization produces identical features for all images, the model instantly collapses.

### 4. Initialization Audit
- The CNN uses default PyTorch initialization (uniform) and contains 6 Convolutional layers with ReLUs and MaxPools, but **zero Batch Normalization layers**. Deep CNNs without BatchNorm are highly prone to covariant shift, activation saturation, and feature homogenization.

## Final Decision Tree Classification
**F. Combination of the above**
- **B. CNN feature extraction issue:** The CNN produces 99.5% identical features for all images.
- **E. Optimization issue (Vanishing Gradients):** The CNN and LSTM gradients die instantly, preventing the CNN from learning to differentiate images.

## Recommended Next Experiment
We must fix the CNN feature homogenization and vanishing gradients. 
**Recommended Action:** Modify `models/recognition/encoder.py` to include `nn.BatchNorm2d` after every `nn.Conv2d` layer. Then, repeat this exact 4-sample overfitting diagnostic to prove that the CNN Cosine Similarity drops to a healthy level (e.g., < 0.5) and the model achieves 0.0 CER on the single batch.
