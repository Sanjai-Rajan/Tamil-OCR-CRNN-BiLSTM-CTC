# Baseline post-evaluation analysis — E1 (official test) and E4 (synthetic validation)

Read-only analysis of the frozen baseline. No training was run, no checkpoint, config, source file or dataset
was modified, and nothing under `runs/baseline_hw/` was touched. Every number below comes from
`reports/eval_test_baseline_summary.json`, `reports/eval_synth_baseline_summary.json`, the two matching
`*_errors.tsv` files, and `reports/baseline_training_complete.md`.

---

## 1. Executive factual summary

- **TEST RESULT (IIIT handwritten test split, 16,184 samples):** CER 0.6006%, WER 4.4550%, word accuracy
  95.5450%, length ratio 0.99945.
- **VALIDATION RESULT (IIIT handwritten validation split, 11,598 samples, from the frozen record):**
  CER 0.5726%, WER 4.5008% (`best.pth`, epoch 36).
- **Synthetic printed-Tamil evaluation (6,000 frozen images, never used in training):** CER 19.1629%,
  WER 47.35%, word accuracy 52.65%, length ratio 0.94325.
- Test and validation agree closely: CER differs by 0.028 percentage points, WER by 0.046 points. The
  ranking of the two even reverses between the metrics (test CER slightly higher, test WER slightly lower).
- The measured difference between the handwritten test set and the synthetic printed set is large: CER is
  31.9× higher on the synthetic set, and 47.35% of synthetic words are wrong against 4.46% of handwritten
  test words.
- For external context, the dataset's own published baseline (Gongidi & Jawahar, ICDAR 2021) reports Tamil
  test CER 1.28% and WER 7.38% on this same test split.

---

## 2. E1 — official test results

Source: `eval_test_baseline_summary.json`, checkpoint `runs/baseline_hw/best.pth`, manifest
`data/real/iiit_hw/manifest_test.jsonl`.

**TEST RESULT — overall**

| Field | Value |
|---|---|
| Samples | 16,184 |
| CER | 0.600638% |
| WER | 4.455017% |
| Word accuracy | 95.544983% |
| Length ratio | 0.999446 |

**TEST RESULT — per length bucket** (bucket = reference length in characters)

| Bucket | n | CER | WER | Word accuracy | Length ratio |
|---|---|---|---|---|---|
| 1–5 | 1,546 | 0.994188% | 3.492885% | 96.507115% | 0.999388 |
| 6–10 | 8,527 | 0.583694% | 3.834877% | 96.165123% | 0.999580 |
| 11–15 | 4,941 | 0.635518% | 5.444242% | 94.555758% | 0.999035 |
| 16+ | 1,170 | 0.431361% | 6.068376% | 93.931624% | 1.000237 |

The summary also repeats every bucket under a `src:manifest_test` key with identical values, because all
samples come from one source.

**Error statistics derived from `eval_test_baseline_errors.tsv`** (the file lists every wrong word:
721 rows, which matches 4.455% of 16,184):

- Total edit operations: 954 — 500 substitutions (52%), 271 deletions (28%), 183 insertions (19%).
- Mean 1.32 edits per wrong word. 596 of 721 wrong words (83%) are wrong by a single edit; 94 by two;
  only 9 by six or more.
- Predicted string shorter than reference in 186 cases (26%), longer in 138 (19%), same length in 397 (55%).
- Zero empty predictions.
- Mean reference length among errors 10.5 characters, mean prediction length 10.4.

---

## 3. E4 — synthetic validation results

Source: `eval_synth_baseline_summary.json`, same checkpoint, manifest `data/synth_val/manifest.jsonl`
(6,000 frozen images: 3,000 modern-print and 3,000 old-print by construction). The baseline training config
contains no `type: synth` source, so none of these images or their seeds were used in training.

**Overall**

| Field | Value |
|---|---|
| Samples | 6,000 |
| CER | 19.162928% |
| WER | 47.35% |
| Word accuracy | 52.65% |
| Length ratio | 0.943251 |

**Per length bucket**

| Bucket | n | CER | WER | Word accuracy | Length ratio |
|---|---|---|---|---|---|
| 1–5 | 915 | 45.794393% | 58.251366% | 41.748634% | 1.000850 |
| 6–10 | 3,165 | 19.058068% | 43.886256% | 56.113744% | 0.939660 |
| 11–15 | 1,556 | 15.806868% | 46.208226% | 53.791774% | 0.942267 |
| 16+ | 364 | 15.186246% | 54.945055% | 45.054945% | 0.929272 |

**Error statistics derived from `eval_synth_baseline_errors.tsv`** (2,841 rows = 47.35% of 6,000):

- Total edit operations: 10,549 — 5,547 substitutions (53%), 4,063 deletions (39%), 939 insertions (9%).
- Mean 3.71 edits per wrong word, against 1.32 on the handwritten test set.
- 903 of 2,841 wrong words (32%) are wrong by a single edit, but 693 (24%) are wrong by six or more edits.
- Predicted string shorter than reference in 1,619 cases (57%), longer in 494 (17%), same length in 728 (26%).
- 3 empty predictions.
- Mean reference length among errors 9.1 characters, mean prediction length 8.0.

**Not available from these files:** the summary is not broken down by synthetic style, so the modern-print
and old-print halves cannot be separated from the current outputs. `src:manifest` covers all 6,000 rows, and
the error TSV carries only reference and prediction columns. Any modern-versus-old statement would be
unsupported by the present evidence.

---

## 4. Validation versus test comparison

| Set | Samples | CER | WER | Word accuracy | Length ratio |
|---|---|---|---|---|---|
| IIIT validation (VALIDATION RESULT) | 11,598 | 0.5726% | 4.5008% | 95.4992% | 0.9996 |
| IIIT test (TEST RESULT) | 16,184 | 0.6006% | 4.4550% | 95.5450% | 0.9994 |
| Synthetic validation | 6,000 | 19.1629% | 47.35% | 52.65% | 0.9433 |

Test minus validation: CER +0.0280 percentage points (+4.9% relative); WER −0.0458 percentage points
(−1.0% relative). Both differences are small and they point in opposite directions, so the test split gives
no indication that model selection on the validation split inflated the validation number.

Note on the sets themselves, from the frozen record and the earlier split investigation: the two handwritten
splits come from the same dataset, share a high proportion of vocabulary with the training split, and
9 byte-identical image groups exist across splits (3 train↔val, 6 train↔test). These properties apply to
validation and test alike, so they do not affect the validation-versus-test comparison, but they do bound
what either number describes.

---

## 5. Handwriting versus synthetic comparison

Measured differences, handwritten test → synthetic set:

- CER 0.6006% → 19.1629% (31.9× higher).
- WER 4.4550% → 47.35% (10.6× higher).
- Word accuracy 95.545% → 52.65%.
- Length ratio 0.9994 → 0.9433, i.e. predictions on the synthetic set are on average about 5.7% shorter than
  the reference, while on the handwritten set the lengths match.
- Mean edits per wrong word 1.32 → 3.71, and the share of wrong words needing six or more edits rises from
  1.2% to 24%.

**Assessment of the proposed statement.** The second half — "has a substantial domain gap on the synthetic
printed-Tamil data" — is directly supported: the same checkpoint, the same preprocessing and the same decoder
produce 0.60% CER on 16,184 handwritten test images and 19.16% CER on 6,000 synthetic printed images that
were not part of training. The first half is supported only in the specific sense that on this dataset's own
test split the model reaches CER 0.6006% and WER 4.4550%, which is lower error than the dataset's published
Tamil baseline (CER 1.28%, WER 7.38%). No subjective rating is asserted here, and nothing in these two
evaluations describes performance on real printed pages, real photographs or scanned documents.

---

## 6. Error analysis

**Handwritten test errors (721 wrong words).**

- Errors are shallow: 83% of wrong words differ from the reference by exactly one edit, and the mean is
  1.32 edits per wrong word.
- Substitutions dominate (52%), and length is preserved in 55% of wrong words. Predictions are not
  systematically shorter or longer, consistent with the overall length ratio of 0.9994.
- The most frequent single edits are, with counts: missing virama ் (84) and spurious virama ் (56);
  missing ு (37) and spurious ு (28); missing க (22); க→த (21); missing ி (21); ள→ன (17); ன→ள (14);
  ண→ன (13); missing ா (12); ீ→ி (12); ஓ→ஒ (12); த→க (11).
- Two groups are visible in that list: diacritic-level errors (virama and vowel signs added or dropped) and
  substitutions between visually similar consonants (ள/ன/ண, க/த) or between short and long vowel forms
  (ீ/ி, ஓ/ஒ).
- Per-bucket numbers show CER lowest for the longest words (0.431% at 16+ characters) while WER rises with
  length (3.49% at 1–5 up to 6.07% at 16+), which is the expected consequence of more characters per word
  when errors are roughly one character each.

**Synthetic errors (2,841 wrong words).**

- Errors are deeper: mean 3.71 edits per wrong word, and 24% of wrong words need six or more edits.
- Deletions account for 39% of edits against 28% on the handwritten set, predictions are shorter in 57% of
  wrong words, and the set-wide length ratio is 0.9433. Predictions are therefore systematically shorter
  than references on this set, which is not the case on the handwritten set.
- The most frequent edits are dominated by dropped characters: missing ் (423), missing க (213),
  missing ு (210), missing த (179), spurious ் (167), missing ி (159), missing ை (138), missing - (127),
  missing ர (123), missing ம (119), missing ா (110), missing ன (107), ௌ→ள (102), missing வ (97).
- The 1–5 character bucket has by far the highest CER (45.79%) while the longer buckets sit between 15% and
  19%, so per-character error is concentrated in short words, whereas word-level error is high everywhere
  (43.9% to 58.3%).

**Qualitative difference.** The two error sets are not the same kind of failure. Handwritten errors are
mostly single-character slips within a correctly-read word of the right length. Synthetic errors frequently
involve several characters at once and a prediction shorter than the reference.

**What the TSVs cannot establish.** They contain no image path, no style label, no font, no degradation
parameters and no confidence values. They therefore cannot show which synthetic style, font, degradation or
text type the failures concentrate in, and they cannot separate genuinely illegible renderings from legible
ones the model misread. Note also that the synthetic set's text sampler includes pseudo-words and numerals
by design, and the error file cannot be used to measure how much of the synthetic error rate those account
for.

---

## 7. What is established

1. The baseline generalises from the validation split to the untouched official test split: CER 0.6006% and
   WER 4.4550% on 16,184 test samples, within 0.03 and 0.05 percentage points of the validation values.
2. There is no length collapse on handwritten data: test length ratio 0.999446, and prediction lengths match
   reference lengths in 55% of the few wrong words.
3. The model handles long handwritten words: the 16+ bucket has the lowest CER of all buckets (0.4314%).
4. On the frozen synthetic printed set, which was never used in training, the same checkpoint reaches CER
   19.1629% and WER 47.35%. A large domain gap between the handwritten training domain and this synthetic
   printed set is demonstrated.
5. Errors on the two sets differ in kind, as quantified in §6.
6. The handwritten test result sits below the published baseline error rates for this dataset and split
   (CER 1.28%, WER 7.38%).

---

## 8. What remains unknown

1. The modern-print versus old-print breakdown of the synthetic result.
2. Whether the handwritten splits are writer-disjoint, and therefore how much of the handwritten result
   depends on writers seen in training (E3).
3. Performance on out-of-vocabulary words, which the published paper reports separately for Tamil
   (CER 1.88%, WER 8.25%) (E2).
4. Performance on real printed pages, real photographs or scanned documents. Nothing measured so far uses a
   real non-handwritten image.
5. What fraction of the synthetic errors comes from pseudo-words, numerals or renderings that are
   illegible in principle.
6. Whether any decoding change, such as lexicon-constrained or beam decoding, would alter either result.
   Both numbers were produced with greedy decoding.
7. Per-writer, per-font and per-degradation error distributions.

---

## 9. Hypotheses for future investigation

These are **hypotheses**, not findings. None is demonstrated by the evidence in §2–§6.

- **H1.** The synthetic gap is mostly explained by the model never having trained on printed glyph shapes.
- **H2.** The gap is concentrated in the old-print half of the synthetic set rather than spread evenly.
- **H3.** Part of the gap comes from synthetic degradations severe enough that the text is not recoverable,
  so the achievable error rate on that set is above zero.
- **H4.** Part of the gap comes from the synthetic text distribution, specifically pseudo-words and numerals,
  which differ from the handwritten training vocabulary.
- **H5.** The high CER in the synthetic 1–5 character bucket reflects short-word behaviour also visible on
  handwritten data, where 1–5 also has the highest CER.
- **H6.** The shorter-prediction pattern on synthetic data reflects the model emitting blanks when the input
  looks unfamiliar.
- **H7.** Font coverage, preprocessing assumptions, augmentation range or model capacity contribute.

Distinguishing H1 from H2, H3 and H4 requires measurement, not reasoning; §10 gives the cheapest next step.

---

## 10. Recommended next diagnostic

**Recommended: split the existing synthetic evaluation by style — modern print versus old print — and then
by text type.** No training, no new data, and the images already exist.

Why this one first:

- It is the largest unexplained quantity in the current evidence. A single number, 19.16% CER, currently
  covers two deliberately different domains, and §3 shows the current outputs cannot separate them.
- It directly discriminates between H2 and the alternatives. If modern print is, say, near 5% CER and old
  print near 33%, the next experiment should target old print. If both halves are close to 19%, the gap is
  general to printed text and a single mixed-print experiment is the right response.
- It also bounds H3 and H4 when combined with a check of which references are pseudo-words or numerals,
  since the synthetic manifest records the label, style and font for every image.
- It decides the design of experiment 001 before any GPU time is spent, and it costs a few minutes.

This needs one small addition to `tools/evaluate.py`, a filter on the manifest's `style` field, equivalent to
the `where` support that `ManifestDataset` already has. That is a tools-level change outside the frozen set,
and it is **not** performed as part of this task.

Secondary candidates, in order: E2 (out-of-vocabulary test subset, which makes the handwritten result
directly comparable to the published 1.88% / 8.25%), E3 (writer overlap between splits, which bounds what
the handwritten result means), and a qualitative look at a sample of synthetic failures alongside their
images to test H3.

---

## 11. Frozen baseline artifacts

Unchanged and unread-except-for-reporting during this analysis:

- `runs/baseline_hw/best.pth`, `runs/baseline_hw/last.pth`
- `runs/baseline_hw/config.yaml`, `history.json`, `history.csv`, `train.log`, `preds_e000.txt`–`preds_e039.txt`
- `configs/baseline_hw.yaml`
- `reports/baseline_training_complete.md`
- `data/real/iiit_hw/manifest_{train,val,test}.jsonl` and the image files
- `data/synth_val/` (6,000 frozen images and manifest)
- All of `src/`

## 12. No experiment started

No training was started or resumed. No configuration was created or edited. No source file was modified. No
checkpoint was written, overwritten or deleted. No synthetic data was added to any training configuration.
`configs/exp001_synth_modern.yaml` was not created.

---

BASELINE REMAINS FROZEN
E1 COMPLETE
E4 COMPLETE
NO NEW TRAINING STARTED
NEXT EXPERIMENT NOT STARTED
