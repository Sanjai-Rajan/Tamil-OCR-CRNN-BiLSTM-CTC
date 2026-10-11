# Experiment 001 — Design Review

Design document only. No training was started, no experiment configuration was created or edited, and no
file under `runs/baseline_hw/`, `configs/`, `data/synth_val/`, `src/` or `tools/` was modified. The existing
evaluation outputs in `reports/` were read, not rewritten. The additional numbers in §2.3 were computed by
read-only scripts in the session scratchpad that join the existing error TSVs to the existing manifests; no
model inference was run for them.

---

## 1. Baseline that Experiment 001 is compared against

| Field | Value |
|---|---|
| Name | Tamil OCR Real-Handwriting Baseline — CRNNv2 + CTC + Width Bucketing |
| Checkpoint | `runs/baseline_hw/best.pth` (epoch 36, selected by IIIT validation CER) |
| Config | `configs/baseline_hw.yaml` (seed 42, 40 epochs, epoch_size 75,736, batch 64, lr 1e-3, OneCycle, AMP, fused AdamW, width bucketing pool 50, max_batch_columns 32,768) |
| Model | CRNNv2, 11,038,277 parameters, 101 classes, from scratch |
| Training data | `data/real/iiit_hw/manifest_train.jsonl` only (75,736 words), `augment: hw` |
| Model selection | IIIT validation CER (`manifest_val.jsonl`, 11,598 words) |
| Decoding | Greedy CTC |
| Measured training time | 5 h 45 min on RTX 4060 Laptop GPU (8 GB) |

The control for Experiment 001 is this frozen checkpoint, evaluated as it already has been. It is not
retrained.

---

## 2. Evidence base

### 2.1 Verified results of the frozen baseline

| Set | n | CER | WER | Word accuracy | Length ratio |
|---|---|---|---|---|---|
| IIIT validation | 11,598 | 0.5726% | 4.5008% | 95.4992% | 0.9996 |
| IIIT test | 16,184 | 0.6006% | 4.4550% | 95.545% | 0.9994 |
| Synthetic combined | 6,000 | 19.1629% | 47.35% | 52.65% | 0.9433 |
| Synthetic modern | 3,000 | 18.51% | 41.43% | 58.57% | 0.96 |
| Synthetic old | 3,000 | 19.82% | 53.27% | 46.73% | 0.93 |

Sources: `reports/baseline_training_complete.md`, `reports/baseline_e1_e4_analysis.md`,
`reports/synthetic_style_diagnostic_results.md`, `reports/eval_test_baseline_*`, `reports/eval_synth_baseline_*`,
`reports/eval_synth_baseline_{modern,old}_run2_*`.

**Repeatability note.** `synthetic_style_diagnostic_results.md` §6 says byte-identity of the repeated
per-style runs was not verified within that report. It was checked in this session: `cmp` shows
`eval_synth_baseline_{modern,old}_run2_{errors.tsv,summary.json}` byte-identical to the earlier
`eval_synth_baseline_{modern,old}_*` files. That report was not edited.

### 2.2 Observations carried forward (from the existing reports)

- O1. Synthetic error is about 32× the handwritten test error in CER and about 10.6× in WER.
- O2. Old print is worse than modern: +1.31 pp CER, +11.84 pp WER, length ratio 0.93 against 0.96.
- O3. The largest style difference is in 1–5-character words (CER 52.61% against 38.69%).
- O4. At 11–15 and 16+ characters the style CER difference is near zero or reversed.
- O5. Synthetic errors are deeper (3.71 edits per wrong word against 1.32) and deletion-heavy (39% of edits).
- O6. Modern print alone is far from the handwritten result.

### 2.3 New read-only measurements made for this review

These come from joining the existing error TSVs to `data/synth_val/manifest.jsonl`,
`data/real/iiit_hw/manifest_{train,test}.jsonl` and `data/text_corpus/tamil_correction_vocabulary.json`.
The joins reproduce the reported per-style CER, WER and length ratio exactly, which validates them.

**(a) Character coverage.** IIIT training labels use 71 of the 100 charset symbols. The characters
``! " ' ( ) , . 0–9 : ; ?`` appear in synthetic validation labels but never in any IIIT training label.

| Subset | n | CER | WER | Length ratio |
|---|---|---|---|---|
| Modern, all | 3,000 | 18.51% | 41.43% | 0.958 |
| Modern, labels using only characters seen in training ("covered") | 2,687 | 15.81% | 34.61% | 0.972 |
| Modern, labels with at least one unseen character ("uncovered") | 313 | 42.17% | 100.00% | 0.833 |
| Old, all | 3,000 | 19.82% | 53.27% | 0.929 |
| Old, covered | 2,713 | 18.03% | 48.32% | 0.942 |
| Old, uncovered | 287 | 37.71% | 100.00% | 0.803 |

Covered subset by word length:

| Bucket | Modern n | Modern CER | Modern WER | Old n | Old CER | Old WER |
|---|---|---|---|---|---|---|
| 1–5 | 391 | 29.74% | 41.94% | 434 | 47.60% | 64.29% |
| 6–10 | 1,427 | 14.15% | 29.92% | 1,380 | 16.82% | 43.77% |
| 11–15 | 707 | 15.55% | 37.06% | 720 | 15.44% | 45.56% |
| 16+ | 162 | 15.99% | 47.53% | 179 | 14.76% | 55.87% |

Every uncovered sample is wrong (600 of 6,000, 10%). The baseline has never received a training target
for those characters, so they cannot be read correctly. With them removed, the modern–old gap does not
shrink. It widens, from +1.31 to +2.22 pp CER and from +11.84 to +13.71 pp WER. The covered-subset modern
WER is still 34.61%.

**(b) Lexical novelty.** 63.2% of IIIT test labels also occur as IIIT training labels. Only 2.8% (modern) and
2.9% (old) of synthetic labels do. On the handwritten test set, the baseline scores:

| IIIT test subset | n | CER | WER |
|---|---|---|---|
| Label also in train | 10,224 | 0.576% | 4.049% |
| Label not in train (OOV) | 5,960 | 0.642% | 5.151% |

On handwriting, novel words cost about 1.1 pp WER. That is not a statement about printed text, but it means
lexical novelty on handwriting is not large enough to be the whole story for the synthetic set.

**(c) Label composition differs by style.** By text type (heuristic, vocabulary membership):

| | Modern | Old |
|---|---|---|
| Vocabulary words | 2,204 | 2,194 |
| Vocabulary word + punctuation | 290 | 300 |
| Tamil strings not in vocabulary (pseudo-words) | 385 | 392 |
| Numbers | 121 | 114 |
| — of which in the 1–5 bucket: numbers | 56 | 82 |
| — of which in the 1–5 bucket: pseudo-words | 75 | 98 |

This follows from the generator: old print uses Tamil numerals for 50% of numbers against 15% for modern.
So the 1–5 bucket in the two styles is not the same mix of text. That confounds O3 as a pure "style" effect.

**(d) Other facts relevant to the design.**

- `real_docs/` is empty. No real printed-Tamil evaluation set exists in the project. Every printed-domain
  measurement available is synthetic, from the same generator code that any synthetic training would use.
- `synth_val` was generated from 15 font families in `assets/fonts`, the same default font set a synthetic
  training source would use. About 73% of its labels are vocabulary words that a training sampler with the
  same vocabulary would also draw.
- The modern degradation path renders 15% of samples on an inverted (dark) background (`degrade.modern_print`).
  The handwriting augmentation (`RealAugment("hw")`) does not invert. Whether this contributes has not been
  measured.

---

## 3. Problem statement

**The problem tested:** the frozen baseline has a large error on synthetic printed Tamil words in both styles,
including on words whose characters it was trained on. Covered-subset CER is 15.81% (modern) and 18.03%
(old), against 0.64% on handwritten OOV words. The baseline has never seen printed glyphs. Experiment 001
should measure how much of this gap closes when printed training examples are the only thing added.

It does **not** test causes internal to the synthetic set (fonts, degradations, traditional letterforms) and
says nothing about real printed documents (§2.3d).

---

## 4. Observations, hypotheses, intervention, outcomes

**Observations** (measured, §2): O1–O6 plus §2.3 (a)–(d).

**Hypotheses** (untested):

- H-F. Part of the gap common to both styles exists because the training data contains no printed material.
- H-A / H-B / H-C. The old–modern difference comes from the old-print degradation pipeline / traditional
  letterforms / the higher Tamil-numeral rate.
- H-D. Some synthetic samples are illegible in principle, putting a floor above zero on achievable error.
- H-E. The short-word effect is a general short-word property. Handwriting 1–5 also has the highest CER.
- H-G. Some error comes from the decoder, via greedy path choices, and not from the model's per-frame scores.
- H-H. Some error comes from image statistics (for example polarity or contrast) and not from glyph shape.
- H-I. Some error comes from label-space coverage (§2.3a). This part is certain for the uncovered 10%.

**Proposed intervention for Experiment 001** (§6): add the project's existing on-the-fly synthetic printed
word generator as a second training source, with everything else unchanged.

**Measurable outcomes** (§7): CER, WER, word accuracy, length ratio and per-bucket CER on IIIT test,
synthetic combined, modern and old, each also reported on the covered subset, with paired comparisons
against the frozen baseline.

---

## 5. Candidate directions (not ranked)

These are listed in the order the request gave them. The order is not a ranking.

### A. Synthetic modern-print augmentation

- **Hypothesis:** H-F restricted to modern print; tests whether modern-only printed data transfers to old print.
- **Data added:** on-the-fly `SynthSource`, `old_print_prob: 0.0`, same fonts and vocabulary as `synth_val`.
- **Code/config change:** a new config only. `build_source` already supports `type: synth`, `vocab` and
  `old_print_prob`. No source change is required to train.
- **Identical:** model, seed, optimiser, schedule shape, batch size, bucketing, IIIT source and augmentation,
  IIIT validation as the only selection set.
- **Deciding metric:** covered-subset CER/WER on modern, and separately on old (transfer).
- **Confounders:** same generator and fonts as the evaluation set (in-distribution gain); new characters
  become learnable (uncovered-subset gain is mechanical); more total training steps if handwriting exposure
  is held constant; synthetic width estimates default to the median in the bucketer (§9).
- **Feasible:** yes, with the current code.

### B. Synthetic old-print augmentation

- **Hypothesis:** H-F restricted to old print; tests transfer from old to modern.
- **Data added:** as A with `old_print_prob: 1.0`.
- **Code/config change:** a new config only.
- **Identical:** as A.
- **Deciding metric:** covered-subset CER/WER on old, then on modern.
- **Confounders:** as A. Old print also bundles several changes at once (letterpress jitter, page curl, scan
  styles, traditional glyphs, numeral rate), so a result cannot be attributed to any single one of them
  (H-A/H-B/H-C remain entangled).
- **Feasible:** yes, with the current code.

### C. Mixed modern + old synthetic training

- **Hypothesis:** H-F for printed text generally, in the proportion the evaluation set uses.
- **Data added:** one `SynthSource` with `old_print_prob: 0.5`, matching `synth_val`'s 50/50 split.
- **Code/config change:** a new config only. The existing `configs/mixed_print_hw.yaml` is **not usable
  as-is**. Its `generator: modern|old` key is not read by `build_source`, so both of its synthetic sources
  would silently use `old_print_prob: 0.3`. It has no `vocab` key, so the text sampler would fall back to
  its built-in 12-word list. Its `data/synth_val/{modern,old}/manifest.jsonl` validation paths do not
  exist, and it has `TBD_FROM_BENCHMARK` placeholders. It also omits `fused_adamw`, `bucket_by_width`,
  `bucket_pool` and `max_batch_columns`, which the baseline uses. It was left untouched.
- **Identical:** as A.
- **Deciding metric:** covered-subset CER/WER on modern and on old, plus IIIT test non-inferiority.
- **Confounders:** as A. It cannot say which style's data produced a gain in which style.
- **Feasible:** yes, with the current code.

### D. Decoding-only improvements

- **D1 — CTC prefix beam search without a language model.**
  - Hypothesis: H-G.
  - Data: none.
  - Code: a new decoder function and an evaluate-only flag in `tools/evaluate.py`. No training.
  - Identical: checkpoint, preprocessing, all data.
  - Metric: the same four evaluation sets, paired against greedy decoding.
  - Confounders: beam width is a free parameter, so it must be fixed before looking at results. It cannot
    fix uncovered characters, because those classes were never trained.
  - Feasible: yes. Inference only, minutes per run.
- **D2 — Lexicon-constrained or corpus n-gram decoding** (the lexicon or n-gram model built from the
  project's own `data/text_corpus`, which is allowed because nothing is pretrained).
  - Hypothesis: part of the error is correctable by word-level priors.
  - Confounder, a severe one: the synthetic sampler draws about 73% of `synth_val` labels from this same
    vocabulary. A lexicon decoder would therefore be partly scored on its own lexicon, and its synthetic
    gain would be inflated. Pseudo-words and numbers (about 17% of labels) would be pushed toward wrong
    dictionary words.
  - Feasible: yes, but it needs new decoding code, and the synthetic numbers would not be interpretable
    without an out-of-lexicon breakdown.

### E. Other interventions using only existing code and data

- **E1 — Print-like handwriting augmentation**, for example polarity inversion or stronger contrast and
  binarisation in a new `RealAugment` mode.
  - Hypothesis: H-H.
  - Data: none added.
  - Code: `src/augment.py` (new mode) plus a config key.
  - Metric: the synthetic sets, especially the covered subset.
  - Confounders: it changes the handwriting training distribution, so IIIT results can move too, and the
    choice of transforms is itself several changes.
  - Feasible: yes.
- **E2 — Test-time polarity normalisation** (invert dark-background crops before inference).
  - Hypothesis: H-H for the inverted modern subset.
  - Code: an evaluate-only flag.
  - Confounder: `preprocess.py` is the single shared path. The change must stay an evaluation flag, or
    training and inference preprocessing would diverge.
  - Feasible: yes, inference only. A cheaper first step is to count the dark-background crops in
    `synth_val`, which is a diagnostic, not an experiment.
- **E3 — Synthetic-only model from scratch** (no handwriting).
  - Hypothesis: H-D. It measures how learnable `synth_val` is at all, which bounds the achievable floor.
  - Metric: the synthetic sets only.
  - Confounder: it is not a controlled change relative to the baseline. It is a reference point.
  - Feasible: yes, but it costs a full training run.
- **E4 — Fine-tune the frozen baseline with synthetic data** (on a copy).
  - Whether this is allowed depends on the constraint "all model learning must remain from scratch". The
    baseline is the project's own from-scratch checkpoint, but fine-tuning from it is not a from-scratch
    run. That is a policy decision for you, not a technical one, so E4 is left out of Experiment 001.
- **E5 — Font-held-out evaluation** (an evaluation design, not an intervention).
  - `SynthWordGenerator` already has `exclude_families`, and every `synth_val` row records `font`.
  - Excluding 2–3 families from synthetic training would split `synth_val` into seen-font and unseen-font
    subsets without regenerating any image. That separates learning the printed domain from memorising
    specific fonts.
  - It needs `build_source` to pass `exclude_families` through, a one-line change in `src/train.py`, which
    is outside the baseline's frozen artefacts. It also needs a `font` filter in `tools/evaluate.py`.
  - Feasible: yes. The trade-off is that it reduces font diversity in training.

---

## 6. Recommended Experiment 001 question

> **H-001.** Adding on-the-fly synthetic printed Tamil words (modern and old in equal proportion, from the
> project's existing generator) to the unchanged handwriting training stream, with handwriting exposure per
> epoch held constant, lowers CER on the charset-covered synthetic validation subsets of **both** styles
> relative to the frozen baseline, while IIIT test CER does not rise by more than a pre-registered margin δ.

Why this question, from the evidence and not from a preference among candidates:

- The measured gap common to both styles is the largest unexplained quantity. Covered modern WER is
  34.61%, against 5.15% on handwritten OOV words (§2.3a–b).
- H-F is the only hypothesis that the evidence says applies to both styles. H-A/B/C concern only the
  smaller old–modern difference.
- It needs no source-code change to train.

This is a hypothesis to be tested, not a predicted result.

### 6.1 Control

`runs/baseline_hw/best.pth`, unchanged, with its existing evaluation outputs.

### 6.2 Treatment: the one isolated change

A new config (proposed name `configs/exp001_mixed_synth.yaml`, **not created**) that is a copy of
`configs/baseline_hw.yaml` differing only in:

| Key | Baseline | Treatment |
|---|---|---|
| `experiment` | `baseline_hw` | `exp001_mixed_synth` (new run directory; `runs/baseline_hw/` untouched) |
| `sources` | `iiit_hw` weight 1.0 | `iiit_hw` weight 0.5 (same path, same `augment: hw`) **+** `synth_print`: `type: synth`, `vocab: data/text_corpus/tamil_correction_vocabulary.json`, `old_print_prob: 0.5`, fonts default `assets/fonts`, weight 0.5 |
| `train.epoch_size` | 75,736 | 151,472, so the expected handwriting draws per epoch stay 75,736 |
| `val` | `iiit_hw_val` (primary) | `iiit_hw_val` (primary, the **only** selection signal) + `synth_val` modern and old as `primary: false`, via `path: data/synth_val/manifest.jsonl` and `where: {style: ...}`, for monitoring only |
| `train.max_hours` | 24 | set above the benchmarked estimate (§10), so the schedule cannot be truncated |

Everything else is identical: model block, seed 42, epochs 40, batch 64, lr, weight decay, warmup, AMP,
fused AdamW, bucketing settings, grad clip, workers.

**Design choice to confirm before creating the config:** holding handwriting exposure constant means the
treatment trains for twice as many steps. The alternative, keeping epoch_size at 75,736, holds steps constant
but halves handwriting exposure. Neither holds both constant. The additive form is proposed because an IIIT
regression then cannot be blamed on less handwriting. The cost is that "more steps" is confounded with
"added synthetic data". The baseline's validation CER plateaued near epoch 36 of 40 (training loss 0.005),
which suggests, but does not prove, that more handwriting-only steps would add little.

### 6.3 Prerequisites (tooling only; none touch the frozen baseline)

1. **Per-sample output in `tools/evaluate.py`.** Add an optional flag that writes every sample's manifest
   index, reference and prediction, so paired statistics are exact and not reconstructed by label. The
   baseline is then re-scored with this flag into **new** files. Existing outputs are kept.
2. **A covered/uncovered subset report**, computed from the per-sample output against the IIIT training
   character set, as in §2.3a.
3. **A seed-range guard.** Training `SynthSource` seeds are `hash((seed_base, epoch, idx)) & 0x7FFFFFFF`,
   a range that **includes** `synth_val`'s seeds (900,000,000–900,002,999 and 950,000,000–950,002,999).
   For about 3M synthetic draws the expected number of collisions is about 8 (6,000 / 2³¹ per draw). A
   collision with matching `old_print_prob` reproduces the validation image exactly. The effect is
   negligible in size but it is a real leak. Fix it by skipping that seed range in the treatment's synthetic
   source, or at minimum by logging and checking collisions after training.
4. **A benchmark of the treatment config**, with `tools/bench_train.py --config <new config>` and
   `tools/synth_preview.py --bench`. This is measurement, not training (§9, §10).

---

## 7. Evaluation

Selected checkpoint: the treatment's `best.pth` by IIIT validation CER, the same rule as the baseline. Each
test-type set is scored **once**, after training, with `tools/evaluate.py`:

| Set | Manifest | Command difference |
|---|---|---|
| IIIT test | `data/real/iiit_hw/manifest_test.jsonl` (frozen) | — |
| Synthetic combined | `data/synth_val/manifest.jsonl` (frozen, not regenerated) | — |
| Synthetic modern | same | `--style modern` |
| Synthetic old | same | `--style old` |

For each set, report n, CER, WER, word accuracy, length ratio and per-bucket CER/WER. For the synthetic
sets, also report the covered and uncovered subsets. Outputs go to new `reports/eval_*_exp001_*` files.

**Paired comparison against the control** (same images, same order):

- WER: McNemar's exact test on per-word correct/incorrect, plus the difference with a 95% CI.
- CER: paired bootstrap over samples (10,000 resamples, fixed seed), giving a 95% CI for treatment minus
  control.

**Pre-registered margin δ (to be fixed by you before training).** Proposed: δ_CER = +0.10 pp and
δ_WER = +0.50 pp absolute on IIIT test. Rationale: both are several times the observed validation–test
differences (0.028 pp CER, 0.046 pp WER). The baseline's run-to-run seed variance is **unknown** (one seed
only), so δ is a declared tolerance, not a measured noise floor.

---

## 8. Interpreting the result (mechanical, no scoring)

Define ΔCER_style = treatment − control on the covered subset, per style, with 95% bootstrap CI, and
ΔCER_IIIT on IIIT test.

| Outcome | Condition | What it means |
|---|---|---|
| H-001 supported | Both style CIs lie entirely below 0 **and** the upper CI bound of ΔCER_IIIT ≤ δ_CER (same for WER) | Adding printed data lowered synthetic error on trained characters in both styles, without a handwriting regression beyond δ |
| Style-asymmetric | One style's CI is below 0 and the other's includes 0 or lies above it | Transfer did not reach both styles; H-001 not supported as stated |
| Synthetic gain with handwriting cost | Style CIs below 0, but the upper bound of ΔCER_IIIT > δ | A trade-off was measured; H-001 not supported |
| No synthetic gain | Both style CIs include 0 or lie above it | H-001 not supported |
| Invalid run | The run hit `max_hours` before the schedule finished, had nonfinite losses, or the synthetic source logged a text-sampler fallback or wrong `old_print_prob` | No conclusion; fix and rerun |

Reported alongside, but not deciding: the uncovered subset (a gain is expected mechanically once those
characters get targets), the length ratio (movement toward 1.0), and covered 1–5-bucket CER per style.

Whatever the outcome, every conclusion concerns **this generator's** synthetic distribution. `synth_val` is
generated by the same code, fonts and vocabulary as the training source (§2.3d). No outcome here is evidence
about real printed documents.

---

## 9. What would justify Experiment 002, and what would cause a rethink

**Evidence that justifies proceeding** (the next question then targets what remains):

- H-001 supported, and a measurable residual remains. Examples: the old–modern covered gap persists, or
  covered 1–5 CER stays well above longer buckets. That points Experiment 002 at H-A/B/C (old-print
  components, one at a time) or at H-E.
- Before any Experiment 002 that tunes anything, such as the synthetic fraction or which degradations to
  use, export a **second frozen synthetic set** with a different seed base, to use as a development set.
  Otherwise `synth_val` becomes a tuning set and stops being a test.

**Evidence that means stop and rethink:**

- No synthetic gain on the covered subsets. First suspect the pipeline, not the hypothesis. Check that the
  synthetic source actually ran with the intended vocabulary and style mix (§5C shows how easily a config
  can silently diverge), and inspect a sample of generated training images.
- A gain only on the uncovered subset. The model learned new characters, not printed glyph shapes.
- A handwriting regression beyond δ. The mixing design, not only the data, needs rethinking before any
  further print work.
- An asymmetric or ambiguous result close to δ. A single seed per arm cannot separate this from seed noise.
  A second-seed run of both arms is needed first. That would be new run directories; the frozen baseline
  would not be retrained in place.

Independent of the outcome: without a real printed evaluation set (`real_docs/` is empty), no claim about
real printed Tamil can be made. Hand-labelling a small set of real printed word crops from your own scans
would meet every project constraint and is the missing evidence.

---

## 10. Expected training cost (from measured numbers only; nothing was run)

Measured:

- Baseline actual run: 40 epochs × 75,736 samples + validation in about 5 h 45 min, about 8.6 min per epoch
  overall.
- `bench2_out.txt`: 236 img/s width-bucketed (28% padding), 57 img/s with random batches (50% padding),
  validation 11,598 words in about 0.8–1.0 min.

**Not measured:** synthetic generation throughput on 4 workers, and bucketing efficiency when synthetic
samples are present.

| Scenario | Basis | Estimate |
|---|---|---|
| Lower bound | Synthetic samples cost the same as handwritten ones; 2× samples per epoch → 2 × 5 h 45 min, plus synthetic monitoring validation (6,000 images per epoch at about 230 img/s ≈ 0.5 min × 40 ≈ 20 min) | **≈ 11.8 h** |
| Bucketing-degraded | `SynthSource.est_width` returns `None`, so the bucketer assigns every synthetic sample the median width. If the synthetic half runs at the measured random-batch rate (57 img/s): 75,736 / 57 ≈ 22 min per epoch for synthetic, plus about 8.6 min for handwriting | **≈ 20–21 h** |
| Generation-bound | If CPU generation with 4 workers is slower than the GPU consumes, time is set by generator speed. Unknown until `synth_preview.py --bench` is run | **unbounded by current evidence** |

Memory risk: `max_batch_columns` relies on width estimates, and synthetic words can reach `MAX_W = 1024`
after resizing. A median-estimated batch can therefore exceed the 32,768-column cap the baseline relied on
to stay within 8 GB (baseline peak 7.2 GB). Prerequisite 4 (the benchmark) must be run before training to
set `max_hours`, confirm memory, and replace this range with a measured figure.

---

## 11. Integrity statement

- `reports/synthetic_style_diagnostic_results.md` already existed (saved 2026-09-23 17:37) with the
  attached content. No attachment arrived with this request, so the existing file was kept unchanged.
- Created: this file only.
- Not modified: `runs/baseline_hw/`, `configs/baseline_hw.yaml`, `configs/mixed_print_hw.yaml`,
  `data/synth_val/`, `data/real/`, `src/`, `tools/`, and all existing `reports/eval_*` outputs.
- No experiment configuration created. No training started. No checkpoint written.

---

BASELINE REMAINS FROZEN
E1 COMPLETE
E4 COMPLETE
STYLE DIAGNOSTIC COMPLETE
EXPERIMENT_001 DESIGN REVIEW COMPLETE
NO TRAINING STARTED
NO EXPERIMENT CHECKPOINT CREATED
