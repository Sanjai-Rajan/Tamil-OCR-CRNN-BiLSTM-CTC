# Experiment 001 — Pre-registration addendum

Written 2026-09-23, **before** Experiment 001 training, before any treatment checkpoint exists and before any
treatment result has been seen. It completes the evaluation procedure of
`reports/experiment_001_design_review.md` §7–§8 and records decisions approved by the project owner. It does
not change the hypothesis, the control, the treatment's single change, the evaluation sets or δ.

---

## 1. Approved decisions

| Item | Value | Status |
|---|---|---|
| δ_CER (IIIT test, treatment − control, absolute) | **+0.10 percentage points** | approved, unchanged from design review §7 |
| δ_WER (IIIT test, treatment − control, absolute) | **+0.50 percentage points** | approved, unchanged from design review §7 |
| `train.unknown_width_estimate` in `configs/experiment_001.yaml` | **736** (replaces 517) | approved |
| `train.max_hours` | 24, **conditional** on the bucketed benchmark of the final configuration measuring ≥ 90 img/s | rule approved; result recorded separately |

δ is a declared tolerance, not a measured noise floor. It is fixed from this point on and may not be changed
after any treatment result exists.

---

## 2. Evaluation procedure (completes design review §7)

**Checkpoint.** `runs/experiment_001/best.pth`, selected during training by IIIT validation CER only (the same
rule as the baseline). Control: `runs/baseline_hw/best.pth`. Each is scored once per set.

**Sets.** IIIT test (`data/real/iiit_hw/manifest_test.jsonl`, 16,184), synthetic combined
(`data/synth_val/manifest.jsonl`, 6,000), synthetic modern (`--style modern`, 3,000), synthetic old
(`--style old`, 3,000). No set is regenerated.

**Per-sample records.** `tools/evaluate.py --samples` writes one row per scored sample carrying the 0-based
line index in the manifest file, the image path as written in the manifest, style, font, reference,
prediction, edit distance and lengths. Control and treatment are paired on `(manifest_index, image, reference)`;
the comparison refuses to run if the two files do not contain exactly the same keys.

**Metrics.** Identical to `src/metrics.py`: CER = Σ edit distance / Σ reference characters (micro-average);
WER = fraction of samples whose prediction is not an exact match; word accuracy = 1 − WER; length ratio =
Σ predicted characters / Σ reference characters. Length buckets by reference length: 1–5, 6–10, 11–15, 16+.

**Covered / uncovered.** The covered character set is the set of characters occurring in the NFC-normalised,
encodable labels of `data/real/iiit_hw/manifest_train.jsonl` (71 characters at the time of writing). A
sample is *covered* if every character of its reference is in that set, otherwise *uncovered*.

**Differences.** Δ = treatment − control, expressed in percentage points.

**Confidence intervals.** Paired percentile bootstrap:
- resampling unit: the paired sample (the same indices are drawn for control and treatment);
- 10,000 resamples, with replacement, of size n (the number of samples in the subset being analysed);
- random generator `numpy.random.default_rng(20260923)`, re-created with the same seed for every
  (set, subset, bucket) analysed, so each analysis is reproducible on its own;
- within each resample, CER is recomputed as a micro-average (Σ edits / Σ reference characters of the
  resampled items), never as a mean of per-sample CERs;
- the 95% CI is two-sided: the 2.5th and 97.5th percentiles of the resampled Δ (numpy default linear
  interpolation);
- the same bootstrap gives the CI for ΔWER.

**McNemar's exact test** (reported, **not** used by the decision rule). b = samples correct in control and
wrong in treatment, c = wrong in control and correct in treatment; two-sided exact binomial p =
min(1, 2 · P[X ≤ min(b, c)]), X ~ Binomial(b + c, ½); p = 1 when b + c = 0.

---

## 3. Decision rule (completes design review §8)

Evaluated mechanically by `tools/compare_eval.py`, in this order of precedence:

1. **Invalid run** — decided from the run record before the comparison is read: training stopped by
   `max_hours` before the 40-epoch schedule finished; any epoch with `nonfinite_losses > 0`; or the logged
   synthetic source differs from the approved configuration (vocabulary not loaded, wrong `old_print_prob`,
   wrong weights or epoch size). An invalid run gets no outcome.
2. Let **S** = the number of synthetic styles (modern, old) whose **covered-subset ΔCER 95% CI upper bound
   is < 0** (strictly). Let **HW_ok** = (IIIT test ΔCER CI upper bound ≤ +0.10 pp) **and** (IIIT test ΔWER
   CI upper bound ≤ +0.50 pp), on all 16,184 test samples.
   - S = 2 and HW_ok → **H-001 supported**
   - S = 2 and not HW_ok → **synthetic gain with handwriting cost**
   - S = 1 → **style-asymmetric** (HW_ok is reported alongside)
   - S = 0 → **no synthetic gain**

Reported but not deciding: uncovered subsets, synthetic combined, length ratios, per-bucket deltas
(especially covered 1–5), McNemar p-values. The design review §9 "stop and rethink" flag **gain only on
uncovered** is raised when S = 0 and at least one style's uncovered ΔCER CI upper bound is < 0.

Known operating characteristic, recorded before any result: with a true zero effect, the WER non-inferiority
condition would still fail with probability ≈ 3% / 15% / 28% if 70% / 50% / 30% of the wrong test words are
shared between the two models (single-model bootstrap SE on the baseline: CER 0.030 pp, WER 0.161 pp). The CI
covers test-sampling noise only; seed-to-seed training variance is not measured (one seed per arm).

---

## 4. Corrected baseline memory and runtime findings

Earlier reports (`experiment_001_prerequisites.md` §3.1 and `experiment_001_design_review.md` §10) state the
baseline's peak GPU memory as 7.2 GB. That figure comes from the benchmark and from the first two epochs only.
`runs/baseline_hw/train.log` shows:

- GPU memory peak (reserved) 7.2 GB up to epoch 2 step 800, then **8.9 GB from epoch 2 step 1000 to the end
  of training**, on an 8.0 GB card; allocator retries 0 throughout;
- throughput 237–255 img/s before that point, and **118–173 img/s (median of all 240 logged windows: 158)
  after it**, for the remaining 37+ epochs;
- training time 326.6 min for 3,029,440 samples: **155 img/s realised average**; validation plus checkpoints
  about 18.6 min in total (~26 s per epoch for 11,598 images).

The timing coincidence is consistent with memory spilling past VRAM into system memory, but the cause is not
proven. The `max_batch_columns` cap did not keep the baseline inside VRAM.

---

## 5. Synthetic width ceiling

`configs/experiment_001.yaml` set `unknown_width_estimate: 517` from a 400-sample benchmark. The 20,000-sample
check (`synth_width_check.txt`) measured model-input widths p50 197, p90 310, p99 441, **max 732 px**, so 517
is **disproven** as a ceiling: 57 of 20,000 samples (0.285%) exceed it, and a simulation of 200,000 synthetic
batches (the sampler rounds 517 up to 544 px, i.e. 60 images per batch) put 10% of them above the
32,768-column cap, by up to 35%.

Approved replacement: **736** (the measured maximum rounded up to the 32-px padding multiple). Synthetic
batches then hold 44 images (44 × 736 = 32,384 columns), and the cap holds for every synthetic width up to
736 px. Widths beyond the measured maximum are rare but not excluded (tail beyond 20,000 samples unmeasured);
the hard limit after preprocessing is 1,024 px.

---

## 6. What this addendum does not change

Hypothesis H-001, control, treatment (one synthetic source, `old_print_prob: 0.5`, weights 0.5/0.5,
`epoch_size: 151472`), evaluation sets, selection rule, δ.
