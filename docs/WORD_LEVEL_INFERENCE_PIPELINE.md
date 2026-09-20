# Word-Level Inference Pipeline

## 1. Why the Original Full-Line Approach Caused Temporal Compression
The original inference pipeline attempted to process full, multi-word document lines directly through the CRNN. The `ocr_balanced_temporal` checkpoint uses an `AspectRatioPreservingResize` transformation with a fixed height of `32`. When a long line (e.g., aspect ratio of 5.0) is resized to height 32, its physical width is compressed to ~160 pixels. Because the CRNN further downsamples width by a factor of 8 (`W/8`), the final sequence length is only ~20 CTC frames. This made it mathematically impossible for the decoder to predict 50-100 characters per line, resulting in severe truncation and garbled text (mode collapse).

## 2. Why Word-Level Inference Matches the Training Data
The model was trained exclusively on a word-level Tamil OCR dataset. The training dataset consists of tightly cropped bounding boxes of individual words with aspect ratios generally between 1.5 and 3.0. By segmenting the document into words *before* OCR inference, we ensure that the aspect ratio of each input image matches the training distribution. A word image resized to height 32 will preserve enough physical width to yield an adequate number of CTC frames for its characters.

## 3. Classical Word Segmentation Method
We implemented a deterministic, classical computer vision method for word segmentation without relying on neural networks:
1. **Grayscale Conversion & Binarization:** The line image is thresholded using Otsu's method to produce a binary map of text vs. background.
2. **Vertical Projection Profile:** We project the binary pixels along the Y-axis to identify columns containing text.
3. **Foreground Components & Gaps:** We identify continuous blocks of foreground pixels (connected components) separated by empty columns (gaps).
4. **Dynamic Thresholding:** A dynamic gap threshold is calculated based on the line height (`int(h * 0.12)`). Gaps smaller than this threshold are considered intra-word character spacing and are merged. Gaps larger than the threshold are classified as word boundaries.
5. **Tight Cropping:** Each segmented word is tightly cropped vertically and horizontally with a small safe margin to remove noise.
6. **Reading Order:** The deterministic projection naturally preserves the left-to-right reading order of the words.

## 4. CRNN Inference
Each segmented word crop is independently passed to the CRNN. The aspect ratio is preserved, yielding an appropriate temporal sequence length `T`. 

## 5. CTC Decoding
The model produces a probability distribution over the vocabulary for each frame. We apply `argmax` across the probability distributions to extract the most likely character sequence. Duplicate characters (e.g., `அஅஅணுணு`) and blank tokens are collapsed using standard CTC decoding.

## 6. Word Ordering
Because the word crops are extracted iteratively from the left-to-right vertical projection array, their spatial sequence perfectly matches their temporal sequence on the line.

## 7. Line Reconstruction
The independently decoded text strings for each word are joined with a single space character (`" ".join(word_predictions)`) to reconstruct the full text of the original document line.

## 8. Document Reconstruction
The reconstructed lines are joined with newline characters (`"\n".join(lines)`) to reconstruct the full document text.

## 9. No Pretrained Models or Language Models Used
This pipeline operates strictly on deterministic pixel projection algorithms and the standalone `ocr_balanced_temporal` CRNN checkpoint. No NLP, mT5, transformers, or external language heuristics were introduced.

## 10. Current Limitations
- **Overlapping/Connected Words:** In some manuscript/historical documents, words are written without clear spatial gaps (e.g., connected by continuous underscores, cursive strokes, or dense noise). The deterministic vertical projection method cannot separate words if there is not a single column of zero white pixels between them. In these cases, multiple words are passed as a single crop, which can trigger the temporal compression bottleneck for that specific block.
- **Noise Sensitivity:** Binarization artifacts can bridge word gaps, causing false merges.
