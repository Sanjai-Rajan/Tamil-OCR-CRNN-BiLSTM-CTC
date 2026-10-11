# Tamil OCR — Operator-Assisted Region OCR (V2)

**V2 is an operator-assisted word-region recognition tool. It is not validated full-page OCR.**

Another name for the same thing: *manual region OCR*.

## What V2 is

You give it a page image and a list of word boxes. It cuts each box out of the page and reads it with the frozen Experiment 002-A word recognizer, one word per box. It then joins the words in the order you gave: by column, by line, and by the order of the boxes inside a line.

It has two parts:

- `pageocr/` — an importable package: the recognizer engine, the box format, the page loader and the exporters. It does not depend on Streamlit.
- `app_page.py` — a Streamlit page on top of the package.

## What V2 is not

- It is not full-page OCR. It does not find text, lines, words, columns or reading order. All of that comes from you, through the boxes.
- It is not validated. No accuracy has been measured for it on any real document. Running it on a page tells you what the model predicts for your boxes, not how often it is right.
- It does not deskew, binarize, denoise or otherwise clean the page.
- It does not correct the output: no dictionary, no language model, no spelling correction.
- It does not read PDFs.
- It does not train, fine-tune, evaluate or change any model, config, dataset, report or research record. `app.py` (V1) is untouched and remains a separate entry point.

## Installation assumptions

Nothing new is installed. V2 uses the project's existing `.venv` and only these packages, all already present: torch, numpy, Pillow, opencv-contrib, streamlit.

It also uses the project's own code, unchanged: `src/preprocess.py`, `src/charset.py`, `src/model/crnn_v2.py`, and the padding constant from `src/data/datasets.py`.

## How to launch

From the project folder:

```
.venv\Scripts\python -m streamlit run app_page.py
```

Then open http://localhost:8501 in a browser. Stop it with Ctrl+C in the terminal.

`.streamlit\config.toml` limits uploads to 20 MB. For a larger scan, raise the limit for one run without editing that file:

```
.venv\Scripts\python -m streamlit run app_page.py --server.maxUploadSize 200
```

## Using the page

1. Upload one page image: PNG, JPG/JPEG or single-page TIFF, grayscale or colour, as scanned.
2. Provide the word boxes (see below).
3. Check the preview: every box is outlined and labelled with its id.
4. Press **Run OCR**.
5. Read the table: thumbnail, box id, recognized text, confidence, clipped flag.
6. Read the assembled text, and download it as TXT or JSON.

Uploading a different page, or changing the boxes, clears the previous result.

### How to provide boxes

All coordinates are **whole pixels of the original image**: `x`, `y` is the top-left corner and `w`, `h` the size. The box covers columns `x` to `x+w-1` and rows `y` to `y+h-1`.

| Field | Meaning |
|---|---|
| `id` | your name for the box; must be unique |
| `x`, `y` | top-left corner, not negative |
| `w`, `h` | width and height, greater than 0 |
| `line_index` | which line the word is on, counted from 0 inside its column |
| `column_index` | which column the line is in, counted from 0 |

Every field is required. A box that leaves the page, a fractional coordinate, a missing field or a repeated id is refused with a message; nothing is rounded or repaired for you.

**JSON file**

```json
{
  "schema_version": 1,
  "page": {"width": 2480, "height": 3508, "sha256": "…"},
  "boxes": [
    {"id": "w001", "x": 212, "y": 340, "w": 188, "h": 61, "line_index": 0, "column_index": 0},
    {"id": "w002", "x": 418, "y": 338, "w": 240, "h": 64, "line_index": 0, "column_index": 0}
  ]
}
```

`page` is optional. If it holds a `sha256`, the app refuses the file for any other page image.

**CSV file** (UTF-8; `schema_version` is repeated in every row)

```
schema_version,id,x,y,w,h,line_index,column_index
1,w001,212,340,188,61,0,0
1,w002,418,338,240,64,0,0
```

**Table on the page.** The *Edit table* tab holds the same fields. Add or change rows and press **Apply table**. The current boxes can be downloaded as JSON or CSV, so a table you typed can be kept and reloaded.

Drawing boxes with the mouse is not part of V2. The optional drawing component was left out; boxes come from a file or the table.

### Preview and coordinates

The preview on screen is a reduced copy of the page (at most 1100 px wide). The page shows the original size, the preview size and both scale factors. Boxes are stored, checked and cropped in original pixels only; the preview is never cropped and never sent to the model. `pageocr.boxes.PreviewScale` is the one place where preview and original coordinates are converted.

If a JPEG carries an EXIF rotation, it is applied when the page is loaded, so the page array is the image as a viewer shows it. Make your boxes on the image as displayed.

## How OCR works

1. `pageocr.imageio.load_page` decodes the file once into a 2-D uint8 grayscale array and records its path, SHA-256, shape and dtype.
2. `pageocr.boxes.crop_from_page` cuts each box from that array: `page[y:y+h, x:x+w]`, no resampling.
3. `pageocr.engine.OcrEngine.recognize_batch` checks every crop, then passes it through the project's existing preprocessing (`src/preprocess.py`: height 48, width up to 1024), the 002-A model, and greedy CTC decoding. Crops of similar width are batched together; results come back in the order the crops went in.
4. `pageocr.export` builds the text and the JSON from the boxes' `column_index` and `line_index`.

The engine accepts only 2-D uint8 grayscale arrays. It raises an error for RGB, RGBA and any other 3-D array, boolean masks, float or 16-bit arrays, and crops with exactly two gray levels (binarized). The page loader refuses 1-bit images and pages with at most two gray levels. A very clean digital image with pure black text on pure white, no anti-aliasing, is therefore refused too.

Each result has:

| Field | Meaning |
|---|---|
| `text` | the raw model prediction |
| `confidence` | see below |
| `width_px` | width of the model input at height 48 |
| `timesteps` | output frames the model used |
| `clipped` | `true` if the box, scaled to height 48, was wider than 1024 px and was squeezed to fit. Such a box is probably more than one word; split it. |

**Confidence is uncalibrated.** It is the mean, over the frames where the model's top class is not the CTC blank, of that class's softmax probability; 0 for an empty prediction. It has not been compared with correctness on any data. Use it only as a flagging heuristic to choose which boxes to look at again. It is not a probability of being correct and must not be quoted as an accuracy.

Using the engine without the page:

```python
from pageocr import boxes, export, imageio
from pageocr.engine import OcrEngine

page = imageio.load_page(r"D:\work\page_012.png")
box_file = boxes.load_json(r"D:\work\page_012_boxes.json")
boxes.validate_boxes(box_file.boxes, page.width, page.height)

engine = OcrEngine(r"runs\experiment_002a\best.pth")          # device="cpu" or "cuda" to force one
crops = [boxes.crop_from_page(page.gray, b) for b in box_file.boxes]
results = engine.recognize_batch(crops)

print(export.to_txt(box_file.boxes, results))
```

## Output format

**TXT.** Words of a line are separated by one space, lines by a newline, columns by one blank line. A box with an empty prediction adds no word; a line with no words is left out.

**JSON.** One record per recognized box, in the order the boxes were supplied:

```json
{
  "schema_version": 1,
  "statement": "V2 is an operator-assisted word-region recognition tool. It is not validated full-page OCR.",
  "confidence_note": "…UNCALIBRATED…",
  "engine_version": "pageocr-2.0.0",
  "checkpoint_sha256": "fb29131e…",
  "page": {"name": "page_012.png", "sha256": "…", "width": 2480, "height": 3508},
  "recognitions": [
    {"box_id": "w001", "x": 212, "y": 340, "w": 188, "h": 61, "line_index": 0, "column_index": 0,
     "text": "…", "confidence": 0.97, "clipped": false, "width_px": 148, "timesteps": 37,
     "page_sha256": "…", "checkpoint_sha256": "fb29131e…", "engine_version": "pageocr-2.0.0"}
  ],
  "skipped": [{"box_id": "w009", "x": 0, "y": 0, "w": 1, "h": 40, "line_index": 3, "column_index": 0,
               "reason": "crop is too small: 1 x 40 px"}]
}
```

`skipped` lists boxes the engine refused, with the reason.

## Checkpoint

| | |
|---|---|
| File | `runs\experiment_002a\best.pth` |
| SHA-256 | `fb29131ed71fe0c89229497d0171b1ff2de4e1e581b4d8c1e09d6fbbaf62d5f1` |
| Model | Experiment 002-A, CRNNv2 (CNN + bidirectional LSTM, CTC), trained locally from scratch |

The engine opens the file read-only, hashes the bytes it is about to load, and refuses to start if the hash is not the one above. It never saves a checkpoint.

## Local-only operation

- Everything runs on this computer, on the GPU if one is available, otherwise on the CPU.
- No external OCR, API or cloud service is used. `.streamlit\config.toml` switches usage statistics off and binds the server to `localhost`.
- The app writes no file. Results leave only through the download buttons.
- Frozen research data is off limits. A path containing `real_docs`, `R1_T4b_review` or `r1_pilot` is refused by the page loader, the box loaders and the engine. An upload whose bytes are identical to a file under `real_docs\` or the R1 review folder is refused by the app. This check is by exact content, so a re-saved or cropped copy would not be caught: do not upload R1 pages.

## Tests

```
.venv\Scripts\python -B -m unittest discover -s tests -p "test_pageocr_*.py"
```

The tests use generated images only. They check behaviour (order, determinism, metadata, refusals, file round trips, coordinate mapping, checkpoint integrity), not accuracy.

## Files

| File | Purpose |
|---|---|
| `pageocr/__init__.py` | version, protected-path guard |
| `pageocr/engine.py` | `OcrEngine`, `Recognition`, the input contract |
| `pageocr/boxes.py` | `Box`, JSON/CSV, validation, `crop_from_page`, `PreviewScale` |
| `pageocr/imageio.py` | `load_page`, preview, box overlay |
| `pageocr/export.py` | `to_txt`, `to_json` |
| `app_page.py` | the Streamlit page |
| `tests/test_pageocr_engine.py`, `tests/test_pageocr_boxes.py` | unit tests |
| `README_PAGE_OCR.md` | this file |

## Not in V2

Automatic line detection, word segmentation, column detection, reading-order inference, deskew, layout preprocessing, historical-page processing, a neural detector, line-level recognition and retraining are later stages. None of them has been started.
