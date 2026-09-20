# Character / Grapheme Box Segmentation Experiment

## Objective
The objective of this experiment is to investigate whether visual segmentation into character/grapheme-like regions is geometrically feasible for the existing Tamil OCR dataset. We aimed to determine if word images can be deterministically decomposed using standard computer vision connected-component and grouping techniques, without the use of neural models or external AI.

## Dataset Source
The dataset format was determined by inspecting `data\tamil_ocr_dataset\imported\tamil\test\packet_001`. 
The format is a JSON Lines (JSONL) manifest file (`manifest.jsonl`) where each line maps an `"image"` relative path to its ground-truth `"label"`. The images are stored as variable-width bounding boxes of single words, typically in grayscale or color RGB.

## Sampling Method
A deterministic sample of exactly **200** word images was selected from the `packet_001` test split using a fixed random seed (`seed=42`). 

## Deterministic Preprocessing
For each image, the pipeline executed:
1. Grayscale conversion.
2. Otsu thresholding with inversion (`cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`) to produce a binary ink mask on a black background.
3. Connected Component labeling with 8-way connectivity.
4. Microscopic noise filtering (discarding components with an area < 5 pixels).

## Connected Component Analysis
Raw connected components were extracted and their bounding boxes and geometric statistics (x, y, w, h, area, centroid) were recorded. 
- **Average raw components/word:** 22.8
- **Median raw components/word:** 15.0

## Candidate Grouping Method
A conservative deterministic grouping algorithm was applied to the raw components to form "candidate regions". 
Components were grouped if their bounding boxes significantly overlapped along the horizontal (x) axis or were separated by less than a very small horizontal distance threshold (3 pixels). This aimed to group detached diacritics, dots, or vertically stacked modifier marks into single candidate regions.

## Visualization Method
Visualizations were generated for all 200 samples and stored in `outputs\diagnostics\character_box_analysis\visualizations\`.
- **Original image:** Basis for drawing.
- **Raw connected components:** Thin blue bounding boxes.
- **Candidate grouped regions:** Thick green bounding boxes, uniquely numbered.
These are strictly labeled as "Candidate Character/Grapheme Regions", not ground-truth boundaries.

## Quantitative Observations
- **Number of samples analyzed:** 200
- **Average raw components/word:** 22.8
- **Average candidate regions/word:** 1.83
- **Words with single connected component (where the entire word merged into 1 candidate):** 169
- **Words with fragmented components:** 79 (where raw components heavily exceeded target tokens)

## Short vs Medium vs Long Words
The dataset distribution in this 200-sample slice:
- **Short words (1–5 tokens):** 15
- **Medium words (6–10 tokens):** 114
- **Long words (11–15 tokens):** 59
- **Very long words (16+ tokens):** 12

**Observation:** As word length increases, the raw component count increases linearly (e.g. up to 40+ raw components). However, the number of *grouped candidate regions* does not scale proportionally. In longer words, the horizontal ink strokes almost always touch, resulting in the entire word fusing into a single massive candidate region. 

## Examples of Fragmentation
In some words, a single complex Tamil character grapheme was split into 3-5 distinct unconnected raw components (e.g., disconnected loops or floating vowel markers), showing severe fragmentation before grouping.

## Examples of Merging
The most dominant failure mode of the segmentation algorithm was extreme merging. In 169 out of 200 words (84.5%), touching adjacent characters caused the candidate grouping algorithm to merge the *entire word* into exactly 1 bounding box.

## Examples of Uncertain Boundaries
In cases where characters did not physically touch, projection valleys often cut right through the middle of legitimate multi-stroke graphemes, meaning a valley does not necessarily correspond to a grapheme boundary.

## Limitations
This experiment relied purely on horizontal overlap and hardcoded gap thresholds. It lacked morphological skeletonization or contour convexity analysis which might better split touching strokes. However, the prevalence of touching characters indicates that simple spatial thresholding is wildly inadequate for this dataset.

## Preliminary Interpretation
Visual, deterministic segmentation into distinct character/grapheme boxes appears **highly unpromising and practically infeasible** for this dataset. The cursive nature of the handwriting/font, combined with standard thresholding, results in nearly all adjacent characters touching horizontally. A character-level bounding box approach would require severe, heuristic-heavy stroke severing which is prone to error.

## Next Experiment
Since deterministic visual segmentation fails due to touching characters, the next logical step is to investigate whether a sequence-to-sequence model (like the existing CRNN) relies on implicit, learned receptive fields rather than hard visual boundaries. A recommended next experiment would be a **Saliency / Gradient Activation Mapping Diagnostic** on the CRNN to see where the network "looks" for specific tokens, bypassing the need for physical bounding boxes entirely.
