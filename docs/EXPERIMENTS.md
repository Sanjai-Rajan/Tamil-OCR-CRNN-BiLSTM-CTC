# Segmentation experiments (generated pages only)

These experiments ask why full-page OCR fails, and whether segmentation changes help. **All pages are generated**: printed Tamil words rendered with the project's fonts, with known text and word boxes. No real document or real handwriting was evaluated. The numbers below do **not** estimate accuracy on real pages.

The raw outputs (page images, per-page results, logs) are not part of this repository; the scripts regenerate them.

## Method

- **Conditions** (`dev_tools/synthetic_conditions.py`): clean, two columns, notebook ruling under the line, ruling through the letters, low contrast, shadow, blur, tight spacing, small print, 3.5° skew, perspective, and a handwriting-style printed font. The slant experiment adds the page types of `tests/slant_fixtures.py` (per-line angles, curved lines, uneven spacing, fanned perspective).
- **Splits:** development seeds and a separate held-out split, 3 pages per condition per split. Every variant reads identical image bytes.
- **Recognizer:** the registered Experiment 002-A checkpoint, run through the application's production functions.
- **Metrics:**
  - words matched one-to-one at IoU ≥ 0.5, plus split, merged and missed words;
  - crop integrity;
  - page CER on the exact text, with and without whitespace;
  - WER;
  - condition A vs B: the generator's exact word box cut from the page (A) against the pipeline's own crop (B), read by the same recognizer.
- **Decision rule, fixed in the scripts before each run:**
  - on development, a variant must lower page CER on a target condition by at least 1 point, while no other condition gets worse by more than 0.5 point CER or 1 point matched rate;
  - the same must then hold on held-out;
  - even a confirmed variant stays off by default.

Scripts:
- `dev_tools/segmentation_experiments.py [--pages 3] [--out <new folder>] [--heldout-base N]`
- `dev_tools/slant_experiments.py [--out <new folder>]`
- `dev_tools/experiment_record.py` (writes a Markdown record from a result folder)

## Results

**Recognition vs segmentation** (rule-removal run v2):

| Crops | Words correct |
|---|---|
| A: the generator's exact box cut from the page, all 12 conditions | 5,262 of 5,262 |
| B: the pipeline's own crops | 4,446 of 5,262 |

Almost all of B's failures are on notebook-ruled pages. There the ruling joins letters and lines into one component, so segmentation, not recognition, is the problem.

**Rule removal** (`exp_rule_removal`; rules-1.1), page CER, development / fresh held-out split:

| | Ruled under the line | Ruled through the letters |
|---|---|---|
| baseline | 30.0 % / 26.9 % | 89.8 % / 90.7 % |
| rule removal | 0.0 % / 0.0 % | 1.3 % / 1.6 % |

There was no regression on the other ten conditions. **Verdict: candidate, kept off by default** until it has been tested on real ruled pages, where ruling may be curved, coloured or broken.

The first version (rules-1.0) failed on one development page, because its text-height estimate was wrong. The fix (rules-1.1) was confirmed on a new held-out split, never seen during development.

**Rejected:**
- *Contrast equalisation (CLAHE):* fixes low-contrast spacing errors but regresses two-column and ruled pages.
- *Wider crop margin:* no gain on any condition.

**Angle-aware line tracking** (`exp_angle_lines`; `pageocr/slant_lines.py`):

| | Development | Held-out |
|---|---|---|
| Lines at mixed angles up to ±3°: page CER, baseline → tracking | 18.1 % → 1.6 % | 51.0 % → 1.5 % |
| Curved lines: CER change | +3.2 points | +5.6 points |
| Ruled lines at mixed angles: CER change | +0.6 points | +4.5 points |
| Uneven spacing: CER change | +1.8 points | none |

The 12 standard conditions were identical in both splits. **Verdict: rejected by the fixed rule, kept off.** Those held-out pages have now been seen, so any revised version needs a new held-out split.

## What these results do not show

- Accuracy on real documents or real handwriting. The fonts are fonts the model was trained with, and the word list overlaps common vocabulary, so the recognition side is not independent of training.
- How real ruling, photographs, shadows or curved pages behave; the simulations are simple.
- Anything about the diagnostic warnings' false-positive or false-negative rates on real pages.

The next step is a real, non-sensitive page with 10–15 human-verified word boxes and transcriptions, compared with `dev_tools/compare_manual_crops.py` (template: `dev_tools/manual_samples_TEMPLATE.json`).
