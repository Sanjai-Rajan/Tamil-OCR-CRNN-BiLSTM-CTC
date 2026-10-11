# Synthetic printed Tamil words (step 4)

Generates modern-print and old-print word images on the fly, with labels that exactly match the image.

## Contents
- `src/charset.py` – v2 character set (Tamil letters, signs, digits, symbols, ASCII digits, punctuation) + label normalisation (NFC).
- `src/synth/text_sampler.py` – frequency-weighted corpus words, long-word boost, pseudo-words, numbers, punctuation.
- `src/synth/fonts.py` – finds Tamil-capable fonts, checks glyph coverage, reads weight/width axes.
- `src/synth/render.py` – correct Tamil shaping (Pillow+raqm, or HarfBuzz fallback when raqm is missing).
- `src/synth/degrade.py` – geometry, crop margins, segmentation artefacts, modern-print and old-print degradation.
- `src/synth/generator.py` – `SynthWordGenerator.sample(seed)` and the PyTorch `SynthWordDataset`.
- `tools/check_synth.py` – environment self-check (run first).
- `tools/synth_preview.py` – labelled preview sheets / speed benchmark.
- `tools/synth_export.py` – freezes a fixed synthetic validation set to disk.
- `assets/fonts/` – 15 SIL-OFL Tamil families (29 faces) with their OFL licence files.

## First run (Windows, inside tamil_ocr_v2)
    .venv\Scripts\pip install -r requirements.txt
    .venv\Scripts\python tools\check_synth.py --vocab data\text_corpus\tamil_correction_vocabulary.json
    .venv\Scripts\python tools\synth_export.py --vocab data\text_corpus\tamil_correction_vocabulary.json --out data\synth_val --n-modern 3000 --n-old 3000

Optional: add Windows' own Tamil fonts (Nirmala UI, Latha, Vijaya) with `--fonts assets\fonts C:\Windows\Fonts`.
These are fine for local training but are not redistributable.

## Old print (step 5)
Enabled automatically for `old_print_prob` of samples (default 30%; `--style old` in the preview tool).
- **Pre-1978 letter forms**: fonts that can draw the traditional ணா றா னா லை ளை னை ணை glyphs are detected
  automatically (Anek Tamil ss01, Hind Madurai dlig, Mukta Malar ss04, Tiro Tamil dlig, Karla Tamil by default).
  The label stays standard Unicode, so the model learns that old and new shapes mean the same text.
- **Letterpress**: each character cluster gets its own baseline offset, spacing jitter and ink strength (hand-set type).
- **Book/scan effects**: page curl, tips of neighbouring lines, ink bleed or starved strokes, aged paper, stains,
  show-through, specks, and archive digitisation styles (bilevel scan, photocopy, microfilm).
- **Tamil numerals** are used for 50% of numbers in old print (15% in modern).
- **Legibility guard**: every degraded crop is checked against its clean ink mask (contrast, ink survival per strip,
  shape overlap, blur, fragmentation). Unreadable crops are regenerated, so heavy effects never create label noise.
