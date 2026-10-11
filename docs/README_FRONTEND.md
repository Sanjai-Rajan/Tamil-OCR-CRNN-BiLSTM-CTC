# Using the application

```powershell
python -m streamlit run app.py
```

Open `http://localhost:8501`. The server listens on this computer only, and usage statistics are off (`.streamlit/config.toml`). Stop it with Ctrl+C.

The app has four tabs: **Single image**, **Batch**, **Full-page OCR**, **About and limits**.

## Sidebar: model

- **Checkpoint:** one of the four registered models (`wordocr/checkpoints.py`). *Experiment 002-A* is the default.
- **Device:** Auto (GPU if available), CPU, or GPU (CUDA).
- **Status:**
  - "Model ready: … SHA-256 verified", or
  - "Model not available", with the reason (for example the checkpoint file is missing; it is never downloaded), and a **Retry loading** button.
- Recognition buttons stay disabled until a model is ready.

## Word OCR

### Single image

1. Upload an image of **one Tamil word** (PNG, JPEG, TIFF or WebP).
2. The preview shows the image as the model receives it: grayscale, height 48 px.
3. Press **Run OCR**. The prediction and an uncalibrated score are shown, and can be downloaded as TXT, JSON or CSV.

**Clear** removes the image and the result.

**Refused inputs:**
- An upload that looks like a **page** or several lines is refused, with a pointer to Full-page OCR.
- A blurred word that is only *suspected* to be several lines can be read anyway by ticking "This image shows ONE word: read it as a word anyway". A page verdict cannot be overridden.
- Binarised (two-gray-level) images are refused: the model was trained on grayscale.
- The project's protected research files are refused by fingerprint (`SECURITY_AND_PRIVACY.md`).

### Batch

Upload several one-word images and press **Run OCR on all**. Each file is read on its own: one that fails or is refused does not stop the others. The results table and the CSV export keep every row.

## Full-page OCR

1. Upload an image of **one printed Tamil page** (PNG, JPEG, TIFF or WebP; not PDF).
2. The page is segmented at once, with no model involved. "Layout found: … block(s) · … line(s) · … word region(s) · text height · skew" and the segmentation overlay (blocks blue, lines green, words red) are shown.
3. Press **Read page (N word regions)**. Each word region is cut from the original grayscale page and read on its own; the text is rebuilt in reading order (words joined by spaces, lines by line breaks, blocks by a blank line).
4. **Download TXT / JSON / CSV**: the text, and every word with its box and score.
5. **Word regions (N)**, **Word crops as the model received them** and **Processing details** show what happened.

**Clear page** resets the tab.

### Segmentation settings (expander)

| Setting | Default | Effect |
|---|---|---|
| Automatic crop margin | on (25 % of text height) | white space added around each word box when its crop is cut |
| Crop margin (pixels) | 6 | used when the automatic margin is off |
| Correct page skew | on (up to 5°) | measures the page angle and rotates the image once |
| Ink threshold sensitivity (Sauvola k) | 0.25 | lower finds fainter ink |
| Smallest word gap (× line height) | 0.16 | higher merges more neighbouring pieces into one word |

**Experimental options, all off by default.** Each is an opt-in for diagnosis, and none is a default because none has been tested on real pages:

- *Remove ruling lines (segmentation):* for notebook pages. A candidate on generated pages.
- *Local contrast equalisation:* rejected on generated pages.
- *Fill ruling in crops (changes model input):* a candidate, but it changes the model's input pixels.

Evidence for each: [EVALUATION_AND_LIMITATIONS.md](EVALUATION_AND_LIMITATIONS.md).

## Diagnostic mode

Tick **Diagnostic mode** before or after reading a page. It describes the run; it never changes the OCR result, the crops or the recognition.

- **Summary:** regions, regions with a warning, lines, empty or failed reads, and warning counts. "What each warning means" explains the 11 heuristic warnings, for example `ink_at_crop_edge`, `rule_line_in_crop`, `very_wide` or `taller_than_line`. Warnings say "look here", not "this is wrong".
- **Line boxes:**
  - *Angle-aware line boxes* (default): each line's direction is estimated from its own letters. Green and blue alternate between lines; magenta with "?" means no reliable angle (too few letters, or lines merged by the segmentation); curved lines are drawn as polylines. A caption gives the global deskew angle and how many lines have a local angle.
  - *Horizontal line bands (as segmented):* the bands the segmentation really used.
- **Overlays:** "Original page, region numbers" (uploaded-image coordinates) and "Crop frame, lines and regions" (the image the crops are cut from). Region numbers match the table; orange marks a region with a warning.
- **Regions table:** every region with ID, block / line / word, status, text, warnings, boxes in both frames, crop size, gap and score. Click a column header to sort; "Only regions with a warning" filters it.
- **Inspect one region:** the exact crop the model read, at actual size; the crop in context (red tight box, blue crop with margin); and the region's record.
- **Lines (N):** per line, its band, forced splits, word-gap threshold and gaps; and the **line geometry** table (angle, its source, reliability, curvature, overlap with the next line).
- **Evidence bundle:**
  - **Download diagnostic bundle (ZIP)**, or **Save bundle to diagnostic_runs/** (never overwrites; `diagnostic_runs/` is ignored by Git).
  - Contents: `metadata.json` (input hash, checkpoint path and hash, settings, code fingerprint), `regions.csv`, `lines.json`, `line_geometry.json`, `page_text.txt`, the exact crops (checked against their recorded hashes), the overlays, an optional copy of the input, `SUMMARY.md` and `SHA256SUMS.txt`.
- **Manual crop comparison:**
  - Enter boxes in uploaded-image coordinates (x1, y1, x2, y2) and, only after checking it yourself, the true text with **confirmed** ticked. Press **Compare A and B**.
  - Condition A is your box cut from the page; condition B is the pipeline's crop of the best-overlapping region. Both are read by the same model call.
  - The result shows crop integrity (intact, clipped, split, merged, missed), predictions, CER where a confirmed reference exists, and a pattern. For example, "manual crops correct, pipeline crops worse" points at segmentation.
  - Unconfirmed references are never scored, and model output is never used as a reference. **Download comparison (JSON)** saves it.
  - The same comparison runs from the command line: `python dev_tools/compare_manual_crops.py --page <image> --samples <samples.json> --out <new folder> [--rule-removal]`, with `dev_tools/manual_samples_TEMPLATE.json` as the template.

## Other interfaces

- `python -m streamlit run app_page.py`: an earlier operator-assisted interface (you supply the word boxes as JSON/CSV or in a table).
- `python -m streamlit run app_page_auto.py`: automatic segmentation with more settings (segmentation diagnostics).

`app.py` is the supported interface.

## What the app does not do

- No PDF input.
- No dictionary or spelling correction: the output is the raw model prediction.
- No calibrated confidence: the score is the mean winning-class probability.
- Only the 100 characters of `src/charset.py` can be output.
- No verified accuracy on real documents ([EVALUATION_AND_LIMITATIONS.md](EVALUATION_AND_LIMITATIONS.md)).
