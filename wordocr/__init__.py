"""
wordocr - single-word Tamil recognition behind the local frontend (app.py). No Streamlit in this package.

    checkpoints   the supported checkpoints, each pinned by the SHA-256 recorded when it was evaluated
    inputs        upload validation and decoding (one image file -> 2-D uint8 grayscale), limits, safe file names
    service       loading a checkpoint through pageocr.engine.OcrEngine and recognizing images, one by one
    export        TXT / JSON / CSV of the displayed results
    guard         refusal of protected research data (frozen R1 data, development pages) before any recognition
    pagedetect    the layout check of word mode: is the upload one word, one line, or a page?
    fullpage      full-page OCR: pageocr.pipeline segmentation -> one crop per word region -> word recognizer ->
                  page text rebuilt from the layout; tested on generated pages only
    page_export   TXT / JSON / CSV of a full-page result

Everything that touches pixels or the model is the project's existing code: pageocr.imageio.load_page_bytes (the
upload decoder), pageocr.engine (input contract, src/preprocess.py, CRNNv2, greedy CTC). Nothing here trains,
evaluates, writes a file or contacts a network service.

SCOPE. The recognizer reads ONE cropped word per image. Its character set has no space, so it cannot separate words:
it cannot read a line or a page in one pass. fullpage therefore segments a page into word regions (classical image
processing from pageocr, no model) and reads each region separately. There is no PDF input.
"""
from pageocr import ROOT  # noqa: F401  (also puts the project root on sys.path for src/)

VERSION = "wordocr-1.1.0"          # 1.1.0 (2026-10-10): page check in word mode, full-page OCR
SCOPE_STATEMENT = ("Reads one cropped Tamil word per image. The model has no space character, so it cannot transcribe "
                   "a line, a paragraph or a page; it does not detect or segment text.")
