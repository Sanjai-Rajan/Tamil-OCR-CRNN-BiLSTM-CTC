# Synthetic style diagnostic — modern print versus old print

Diagnostic report only. No training was started, no experiment was created, and the frozen baseline,
`runs/baseline_hw/`, `configs/baseline_hw.yaml` and the synthetic images were not modified or regenerated.

---

## 1. Purpose

The combined synthetic evaluation (E4) reported a single figure, CER 19.1629% and WER 47.35%, over all
6,000 frozen synthetic images. That set is built from two deliberately different domains — 3,000 modern-print
and 3,000 old-print renderings — and the combined output could not be separated by style, so the earlier
analysis (`reports/baseline_e1_e4_analysis.md`, §3 and §10) recorded the split as unavailable and named it as
the next diagnostic.

This report records the result of evaluating the **same frozen checkpoint** separately on the two labelled
subsets. Its purpose is to establish whether the measured synthetic gap is evenly distributed across the two
styles or concentrated in one of them. It does not attempt to explain the cause.

---

## 2. Evaluation identity

| Field | Value |
|---|---|
| Checkpoint | `runs/baseline_hw/best.pth` (frozen baseline, epoch 36) |
| Manifest | `data/synth_val/manifest.jsonl` |
| Modern-print samples | 3,000 |
| Old-print samples | 3,000 |
| Combined | 6,000 |
| Mode | Inference only — no gradient updates, no checkpoint writes |
| Decoding | Greedy CTC, unchanged from the baseline evaluation |
| Training data used by this checkpoint | IIIT handwritten words only; no synthetic data |

The images are the frozen set produced before baseline training, with generator seeds far outside the
training range. None of them, in either style, was seen during training.

---

## 3. Main results

| Subset | Samples | CER | WER | Word accuracy | Length ratio |
|---|---|---|---|---|---|
| Modern print | 3,000 | 18.51% | 41.43% | 58.57% | 0.96 |
| Old print | 3,000 | 19.82% | 53.27% | 46.73% | 0.93 |
| Combined | 6,000 | 19.16% | 47.35% | 52.65% | 0.94 (0.9433 as reported by the original combined evaluation) |

For reference, the same checkpoint on handwritten data: IIIT test CER 0.6006%, WER 4.4550%.

---

## 4. Word-length breakdown

| Bucket | Modern CER | Old CER | CER difference (old − modern) | Modern word accuracy | Old word accuracy | Word-accuracy difference |
|---|---|---|---|---|---|---|
| 1–5 | 38.69% | 52.61% | +13.92 pp | 51.59% | 32.63% | −18.96 pp |
| 6–10 | 18.46% | 19.68% | +1.22 pp | 62.15% | 49.87% | −12.28 pp |
| 11–15 | 15.83% | 15.79% | −0.04 pp | 57.49% | 50.13% | −7.36 pp |
| 16+ | 15.89% | 14.55% | −1.34 pp | 48.02% | 42.25% | −5.77 pp |

Per-style sample counts for each bucket were not recorded in the diagnostic output. The combined evaluation
gives the bucket sizes over all 6,000 images: 915 (1–5), 3,165 (6–10), 1,556 (11–15) and 364 (16+).

---

## 5. Factual observations

1. Old-print CER is **1.31 percentage points** higher than modern-print CER (19.82% against 18.51%), a
   relative increase of about 7.1%.
2. Old-print WER is substantially higher: 53.27% against 41.43%, a difference of 11.84 percentage points.
3. Old-print word accuracy is **11.84 percentage points** lower (46.73% against 58.57%), the same quantity as
   the WER difference expressed in the complementary direction.
4. Old-print length ratio is lower, 0.93 against 0.96, meaning predictions on the old subset are on average
   shorter relative to their references than on the modern subset. Both are below the handwritten test value
   of 0.9994.
5. The largest CER difference between the styles occurs in the 1–5 character bucket: 52.61% against 38.69%,
   a gap of 13.92 percentage points.
6. In the two longest buckets the CER difference reverses sign: old print is marginally **lower** than modern
   print at 11–15 (−0.04 pp) and 16+ (−1.34 pp). The styles are therefore close at these lengths rather than
   showing an old-print penalty.
7. Word accuracy is lower for old print in **every** bucket, and the gap narrows monotonically as words get
   longer: 18.96, 12.28, 7.36 and 5.77 percentage points.
8. Taken together, points 5 to 7 mean the old-style difficulty measured here is concentrated in word-level
   correctness and in short words, not in per-character accuracy on long words.
9. Both styles remain far from the handwritten results. Modern-print CER of 18.51% is about 31 times the
   handwritten test CER of 0.6006%, and modern-print WER of 41.43% is about 9 times the handwritten test WER
   of 4.4550%. The style split therefore shows that the measured synthetic gap is **not** confined to the old
   subset.

---

## 6. Consistency checks

**Error-count arithmetic — verified against the combined evaluation.**

| Quantity | Value |
|---|---|
| Modern wrong words | 41.43% × 3,000 = 1,242.9 → **1,243** |
| Old wrong words | 53.27% × 3,000 = 1,598.1 → **1,598** |
| Sum | **2,841** |
| Combined evaluation | 47.35% × 6,000 = 2,841 wrong words |
| Combined error TSV | 2,841 rows |

The two subsets account for exactly the number of wrong words reported by the combined evaluation, and for
exactly the number of rows in `eval_synth_baseline_errors.tsv`. The word-accuracy figures agree by the same
arithmetic: 58.57% and 46.73% over 3,000 samples each give 52.65% over 6,000.

**Combined CER consistency.** Micro-averaged CER is weighted by reference characters rather than by sample
count, so the combined value is not required to be the exact mean of the two subsets. The observed combined
value, 19.16%, lies between 18.51% and 19.82% and is close to their midpoint of 19.165%, which is consistent
with the two subsets carrying similar total character counts.

**Repeatability.** Both evaluations were run twice. Byte-for-byte identity of the repeated outputs is
**not verified within this report**, because the output files and their checksums are not included in the
material this report is written from. Recording a hash of each summary and error file, for example with
`Get-FileHash`, would make that claim checkable; until then it should be stated as observed by whoever ran
the evaluations rather than as verified here. Note separately that the evaluation is inference-only with
greedy decoding on a fixed image set, so repeated runs are expected to agree, subject to ordinary GPU
floating-point non-determinism.

---

## 7. Interpretation discipline

This diagnostic establishes one thing: a **measured performance difference between two labelled synthetic
subsets**, produced by one unchanged checkpoint under identical inference settings.

It does **not** establish any cause. No claim is made here that old fonts, pre-1978 letterforms, script
reform, ink or paper degradation, page curl, archive-scan simulation, rendering, letterpress jitter, text
sampling or any other specific property produced the difference. The two subsets differ in many ways at once,
and nothing in these numbers isolates any one of them.

The following remain **hypotheses**, untested:

- **H-A.** The old/modern difference is driven by the degradation pipeline applied to the old subset.
- **H-B.** It is driven by the traditional letterform features enabled in part of the old subset.
- **H-C.** It is driven by the different Tamil-numeral probability between the two styles.
- **H-D.** Part of the old subset is illegible in principle, placing a floor above zero on its error rate.
- **H-E.** The short-word effect in observation 5 is a property of short words generally; the handwritten
  test set also has its highest CER in the 1–5 bucket.
- **H-F.** The gap shared by both styles, observation 9, reflects the absence of any printed material in the
  training data.

Separating these requires further measurement, not argument.

---

## 8. Next-step implication

The results provide evidence that **old-print performance deserves targeted investigation in the next
experiment**, since old print is worse than modern print on WER, word accuracy and length ratio, and markedly
worse on short words.

They equally show that modern print is far from the handwritten result, so any investigation scoped to old
print alone would leave the larger shared gap unmeasured.

No experiment design is declared here, no configuration was created, and nothing was trained or modified.

---

## 9. Preserved baseline conclusions

Unchanged by this diagnostic:

- **IIIT test (TEST RESULT):** CER 0.6006%, WER 4.4550%, word accuracy 95.5450%, 16,184 samples.
- **IIIT validation (VALIDATION RESULT):** CER 0.5726%, WER 4.5008%, 11,598 samples, `best.pth`, epoch 36.
- **Synthetic combined:** CER 19.1629%, WER 47.35%, word accuracy 52.65%, 6,000 samples.
- The baseline remains frozen: `runs/baseline_hw/`, `configs/baseline_hw.yaml` and
  `reports/baseline_training_complete.md` are unmodified.

---

BASELINE REMAINS FROZEN
E1 COMPLETE
E4 COMPLETE
STYLE DIAGNOSTIC COMPLETE
NO NEW TRAINING STARTED
EXPERIMENT_001 NOT STARTED
