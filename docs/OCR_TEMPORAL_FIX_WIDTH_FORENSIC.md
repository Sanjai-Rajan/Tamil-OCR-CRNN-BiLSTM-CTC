# OCR Temporal Fix Width Forensic

## Actual Input Pipeline
The input image is processed through `AspectRatioPreservingResize(32)` located in `scripts/dataset_loader.py`.
- **Input Image Height:** Hardcoded to 32 pixels.
- **Input Image Width Strategy:** The aspect ratio is strictly preserved.
- **Resizing/Padding:** The original image is resized using `new_w = max(4, round(w * 32 / h))`. The width is *dynamic* and NOT capped. During batching, shorter words are padded with white space to match the `max_width` of that specific batch.

## Actual CNN Pooling
The CNN architecture in `models/recognition/encoder.py` currently forces standard max pooling that ignores the `less_downsample` parameter passed to it.
Every single pooling layer is defined as:
- Pool 1: `nn.MaxPool2d((2, 1))` (Height / 2, Width / 1)
- Pool 2: `nn.MaxPool2d((2, 1))` (Height / 2, Width / 1)
- Pool 3: `nn.MaxPool2d((2, 1))` (Height / 2, Width / 1)

## Actual Temporal Dimension
Because the width stride is exactly 1 at every pooling layer, the spatial width is **never downsampled**.
- **Exact CNN Output Height:** 32 / (2 * 2 * 2) = 4 pixels.
- **Exact CNN Output Width:** Exact same as the preprocessed image width `W`.
- **Exact T entering BiLSTM:** $T = W$ exactly.
- **Architecture Base:** The architecture operates at $T = W$.

**Concrete Example (Long Word - "கம்பெனிகளிடமிருந்து"):**
- **Original Size:** 784x100
- **Preprocessed Width:** $784 \times (32/100) = 251$
- **CNN Output Width:** 251
- **Exact T:** 251

## Checkpoint Verification
A Python diagnostic check on `checkpoints\recognition\tamil\ocr_temporal_fix\best.pth` confirms that the loaded model weights perfectly match the `T=W` CNN architecture (all max pooling layers are stateless and seamlessly loaded). The checkpoint config and weights are fully consistent with this $T=W$ logic.

## Identified Bottleneck
The hypothesis that the bottleneck was $T = W/8$ or that $T$ was insufficient for CTC ($T < 2N+1$) is **incorrect**. 
A diagnostic test on 1,000 representative TEST samples verified that `Insufficient T` occurred in **0%** of cases. For long words (16-20+ tokens), the average $T$ was 176.5—massively higher than the minimum CTC requirement.

**The TRUE Bottleneck:**
Because width is never pooled, the CNN's receptive field across the horizontal axis is extremely narrow (just 13 pixels total across all 6 convolutions). 
When an image is 250 pixels wide, $T = 250$. The BiLSTM is forced to integrate 250 individual timesteps, each representing a tiny 13-pixel sliver of the image. 
LSTMs severely degrade over sequences longer than ~80-100 timesteps. The BiLSTM successfully aligns the first 4-5 characters, but its hidden state memory degrades over the remaining 200 timesteps, causing it to collapse into predicting blanks. **The truncation is caused by excessive sequence length ($T$) and insufficient CNN receptive field per timestep.**

## Proposed Next Experiment
**Experiment:** Return to a balanced temporal downsampling of $T = W/4$.
**Method:** Modify `models/recognition/encoder.py` so the three pooling layers are:
1. `(2, 2)`
2. `(2, 2)`
3. `(2, 1)`

**Why:**
- A 250px wide image will yield $T = 62$ instead of 250.
- $62 \ge 2(19)+1 \rightarrow 62 \ge 39$. CTC remains perfectly feasible mathematically.
- The LSTM sequence length drops from 250 to 62, which is highly manageable and prevents state degradation.
- The CNN receptive field per timestep quadruples, giving the BiLSTM much richer spatial features.

*Note: This read-only forensic check does not alter the codebase or run this experiment.*
