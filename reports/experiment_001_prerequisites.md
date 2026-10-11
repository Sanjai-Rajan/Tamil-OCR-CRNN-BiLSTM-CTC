# Experiment 001 — prerequisites

Prerequisite check only. **No training was started, no experiment configuration was created, no checkpoint was
written, and nothing under `runs/baseline_hw/`, `configs/`, `data/synth_val/` or `data/real/` was modified.**
One new read-only audit script was added (`tools/bench_synth.py`); it writes no file and trains nothing. No
change was made to `src/`. Proposed source changes are described but **not implemented**.

Inputs read: `reports/experiment_001_design_review.md`, `reports/baseline_training_complete.md`,
`reports/synthetic_style_diagnostic_results.md`, plus the source files listed in §4.

---

## 1. Prerequisite checklist

| # | Prerequisite | Status | Where |
|---|---|---|---|
| 1 | Synthetic-data training throughput and memory; bucketing safety | **NOT MEASURED — needs a run on the RTX 4060.** The mechanism is analysed and quantified; the measurement script is provided | §3.1, §5 |
| 2 | Seed mechanism versus the frozen 6,000-image validation set | **MEASURED EXACTLY** by deterministic simulation: 4 seed collisions, 3 of which would reproduce a validation image | §3.2 |
| 3 | Generator and configuration path for the 50/50 treatment | **VERIFIED from source**; `configs/mixed_print_hw.yaml` confirmed unusable; a silent failure mode found | §3.3, §5 (I2) |
| 4 | Handwriting exposure per epoch preserved | **VERIFIED analytically and by simulation**: mean 75,746 draws/epoch, sd 175, against the baseline's 75,736 | §3.4 |
| 5 | All required paths exist | **PARTIALLY VERIFIED.** Confirmed present from prior reports; a machine-side check is in the script (section A) | §3.5 |
| 6 | Training-cost estimate from measured numbers | **ESTIMATE ONLY.** Synthetic generation throughput and mixed-loop throughput are not measured yet; no invented figure is given | §6 |
| 7 | Regression margin δ recorded | **RECORDED** as a pre-registered design parameter, unchanged, not a validated noise floor | §7 |
| 8 | From-scratch constraint | **VERIFIED from source**: `src/train.py` can only load `runs/<experiment>/last.pth`, and only with `--resume`. The baseline checkpoint is never read by the trainer | §3.6 |

---

## 2. Commands and tests executed

All of the following were run against the same project source files, in an analysis environment. **No GPU
work was performed and nothing in the project was modified.**

| Test | What it did |
|---|---|
| Source reading | `src/train.py` (`build_source`, `make_loader`, `TrainStep`, resume path), `src/data/datasets.py` (`SynthSource`, `MixtureDataset`, `WidthBucketSampler`, `collate`), `src/synth/generator.py`, `src/synth/text_sampler.py`, `tools/synth_export.py` |
| Deterministic draw simulation | Reproduced `MixtureDataset.locate` + `SynthSource.get` seed arithmetic for 40 epochs × 151,472 draws (6,058,880 draws, 69 s). No image generated |
| Style-match check | For each colliding seed, evaluated `random.Random(seed).random()` against `old_print_prob`, which is the draw that `SynthWordGenerator.sample()` uses to choose the style |
| Vocabulary fallback test | Constructed `TextSampler` with a non-existent path and counted loaded words |
| Pickle test | `pickle.dumps` / `loads` of a built `SynthSource`, then generated a sample from the round-tripped object (this is what Windows `spawn` workers do) |
| Fork-safety test | Built a synthetic source, rendered samples in the parent process, then started DataLoader workers |
| Script self-test | `tools/bench_synth.py` end-to-end on a stand-in dataset, all sections |

**Not executed:** anything requiring the project's GPU, the real `data/` tree, or the real vocabulary file.

---

## 3. Results

### 3.1 Prerequisite 1 — throughput, memory, bucketing safety

**Not measured.** It requires the RTX 4060 and the real data. Run, when authorised:

```
.venv\Scripts\python tools\bench_synth.py --vocab data\text_corpus\tamil_correction_vocabulary.json > bench_synth_out.txt 2>&1
```

The script trains nothing and writes no project file. It reports: required paths (A), the generator and text
sampler actually in effect (B), synthetic generation throughput and the width distribution of synthetic
images at the model's input height (C), the bucketing-safety comparison (D), the mixed 50/50 loop with
throughput, GPU memory, allocator retries, padding share and the realised source mix (E), the seed audit (F)
and a cost estimate from the measured loop (G).

**What is already established about bucketing, from the source:**

- `SynthSource.est_width()` returns `None`. `WidthBucketSampler._widths()` replaces every `None` with the
  **median of the known widths**, which are the real handwritten images.
- `max_batch_columns` is therefore enforced against that median estimate, never against the real width of a
  synthetic image.
- A batch of 64 synthetic images estimated at the handwriting median but actually near `preprocess.MAX_W`
  (1,024 px) would occupy 64 × 1,024 = **65,536 columns against a 32,768 cap, twice the intended bound**.
  The baseline relied on that cap to stay inside 8 GB (its measured peak was 7.2 GB).
- Whether this occurs in practice depends on the real synthetic width distribution, which section C measures.
  This is the single most important number to obtain before training.

### 3.2 Prerequisite 2 — seed collisions with the frozen validation set

**Mechanism, from the source:**

- Frozen validation seeds (`tools/synth_export.py`): `seed_base + offset + i`, i.e. 900,000,000–900,002,999
  (modern, exported with `old_print_prob = 0.0`) and 950,000,000–950,002,999 (old, exported with
  `old_print_prob = 1.0`). 6,000 values.
- Training seeds (`SynthSource.get`): `hash((seed_base, epoch, idx)) & 0x7FFFFFFF`, uniform over
  0 … 2³¹−1, where `seed_base = (config seed + source index) × 7919` (`build_source`). The range **contains**
  the validation seeds.
- `SynthWordGenerator.sample(seed)` is a pure function of the seed: it reseeds the text sampler as
  `random.Random(seed ^ 0x5F3759DF)`, so a repeated seed reproduces the same text, font, size, degradation
  and image — **provided** the vocabulary file, font set and `font_size_range` match, and provided the style
  branch matches, because `old = rng.random() < old_print_prob` is the first draw from `random.Random(seed)`.

**Measured exactly** for the design-review treatment (seed 42, 40 epochs, epoch_size 151,472, one synthetic
source with `old_print_prob: 0.5`, weights 0.5/0.5), by simulating every draw:

| Quantity | Value |
|---|---|
| Synthetic draws over the run | 3,029,020 |
| Expected collisions by chance | 8.46 |
| **Exact seed collisions** | **4** |
| … of which reproduce a validation image exactly | **3** (0.050% of the 6,000-image set) |
| Repeated synthetic seeds within the run (duplicate training images) | 111,840 of 3,029,020 (3.7%) |

| Epoch | Seed | Validation style | Style the treatment would draw | Result |
|---|---|---|---|---|
| 2 | 900002793 | modern | modern | identical image |
| 4 | 900000951 | modern | modern | identical image |
| 27 | 950001650 | old | modern | different image |
| 33 | 950002536 | old | old | identical image |

These four values are deterministic, so they can be confirmed on the project machine: section F of the script
prints exactly this table. Python's hash of an integer tuple is version-stable from 3.8 onward, but the
project's Python is 3.12 and this simulation ran on 3.11, so **treat the table as a prediction until section F
confirms it**.

**Smallest safe ways to remove the leak** (none implemented; see §5 I4 for the exact diff):

1. **Seed-range guard, one line in `src/data/datasets.py`** — modulo the training seed into a range that
   excludes the validation band. Removes the leak by construction, for this and every future synthetic run.
   Requires approval because it touches `src/`.
2. **Zero-code option** — accept and document the 3 images. The contamination is 0.05% of a monitoring-only
   set that is not the selection signal. This is defensible but must be stated in the experiment record.
3. **Zero-code option** — use a different vocabulary file for training than `synth_val` used. Different text
   makes collisions impossible, but it changes the treatment's data distribution, so it is not a clean
   isolated change.

### 3.3 Prerequisite 3 — generator and configuration path

**Style selection.** `old_print_prob` is read per source by `build_source` (`spec.get("old_print_prob", 0.3)`)
and used in `SynthWordGenerator.sample()` as the very first random draw. Two ways to reach 50/50:

- one source with `old_print_prob: 0.5` — the split is binomial around 50%, not exact;
- two sources with `0.0` and `1.0` at equal weight — the split is exact, and each source is separately named
  in the logs. This also makes the per-source seed streams independent.

**Vocabulary and text sampling.** `TextSampler` loads `{word: frequency}` JSON, keeps entries with
`freq ≥ 2` that are encodable and ≤ 40 characters, then samples with the fixed mix (0.50 frequent, 0.20
long-tail with a length boost, 0.15 pseudo-words, 0.05 numbers, 0.10 punctuated). `tamil_digit_prob` is set
per sample to 0.5 in old style and 0.15 in modern, which is the confound already recorded in design review
§2.3c.

**`configs/mixed_print_hw.yaml` is unusable as-is — confirmed independently.** `build_source` reads only
`type`, `name`, `path`, `augment`, `where`, `vocab`, `fonts`, `old_print_prob` and `length`. Its
`generator: modern|old` key is **not read by any code path**, so both of its synthetic sources would silently
run at the default `old_print_prob: 0.3`. It also has no `vocab` key (see I2), non-existent validation paths,
`TBD_FROM_BENCHMARK` placeholders, and omits `fused_adamw`, `bucket_by_width`, `bucket_pool` and
`max_batch_columns`. It was not modified and must not be used.

### 3.4 Prerequisite 4 — handwriting exposure per epoch

Analytically: `MixtureDataset` draws each of `epoch_size` samples independently, choosing a source with
probability proportional to its weight. Expected handwriting draws per epoch = `epoch_size × w_hw`. The
baseline is 75,736 × 1.0. The treatment is 151,472 × 0.5 = 75,736, so exposure is preserved **in
expectation**, with binomial variation of sd = √(151,472 × 0.5 × 0.5) ≈ 195.

Measured over the simulated 40 epochs:

| Quantity | Treatment | Baseline |
|---|---|---|
| Handwriting draws per epoch | mean 75,746, sd 175, min 75,426, max 76,135 | 75,736 exactly |
| Handwriting draws over the run | 3,029,860 | 3,029,440 (100.01%) |
| Unique handwritten images touched over the run | 75,736 of 75,736 (100%) | 75,736 of 75,736 |

Exposure is therefore preserved to within about ±0.25% per epoch. Two properties carried over from the
baseline, unchanged: sampling is with replacement, so a single epoch touches about 63% of unique images; and
"epoch" means 75,736 handwriting draws, not a permutation.

**The cost of preserving exposure**, as design review §6.2 already flags: the treatment runs twice the steps
per epoch, so "more steps" is confounded with "added synthetic data". That is a design decision, recorded in
§8 as needing explicit confirmation.

### 3.5 Prerequisite 5 — required paths

Needed by the proposed treatment, with no placeholder or fake path created:

| Path | Purpose | Status |
|---|---|---|
| `data/real/iiit_hw/manifest_train.jsonl` | handwriting source | confirmed by the baseline run |
| `data/real/iiit_hw/manifest_val.jsonl` | selection signal | confirmed by the baseline run |
| `data/text_corpus/tamil_correction_vocabulary.json` | synthetic text sampler | confirmed by design review §2.3 |
| `assets/fonts` | synthetic fonts (15 families) | confirmed by design review §2.3d |
| `data/synth_val/manifest.jsonl` | monitoring validation, `where: {style: …}` | confirmed by E4 and the style diagnostic |

Section A of the script re-checks all five on the project machine and prints sizes. `real_docs/` is empty and
is **not** required by this experiment.

### 3.6 Prerequisite 8 — from-scratch constraint

Verified in `src/train.py`: the model is always built fresh by `build_model`, and the **only** checkpoint read
is `runs/<experiment>/last.pth`, and only when `--resume` is passed. With a new `experiment:` name, no path
exists that could load `runs/baseline_hw/best.pth` into training. The baseline checkpoint is used only by
`tools/evaluate.py` for the control comparison. No fine-tuning path exists or is proposed.

---

## 4. Files inspected

| File | What was checked |
|---|---|
| `src/train.py` | `build_source` key handling, `make_loader`, `make_sampler`, `TrainStep`, optimiser/schedule construction, the resume path, which checkpoints can be read |
| `src/data/datasets.py` | `SynthSource.get` seed arithmetic, `est_width` returning `None`, `MixtureDataset.locate` draw order, `WidthBucketSampler._widths` median substitution and `max_columns` enforcement, `collate` padding |
| `src/synth/generator.py` | `sample(seed)` determinism, style draw position, per-style `tamil_digit_prob`, font picking, legibility retry and fallback |
| `src/synth/text_sampler.py` | vocabulary loading, filters, the sampling mix, **the silent fallback list** |
| `tools/synth_export.py` | how the frozen validation seeds were produced |
| `configs/mixed_print_hw.yaml` (read only) | which keys the code actually reads |
| `configs/baseline_hw.yaml` (read only) | the keys the treatment must copy unchanged |
| `reports/experiment_001_design_review.md`, `baseline_training_complete.md`, `synthetic_style_diagnostic_results.md` | design parameters, frozen results |

---

## 5. Issues found

**I1 — `configs/mixed_print_hw.yaml` would train something other than what it appears to describe.**
Independently confirmed (§3.3). Severity: high if used. Action: do not use; the treatment needs a fresh config.

**I2 — The text sampler fails silently on a bad vocabulary path.** `TextSampler.__init__` uses
`if vocab_json and Path(vocab_json).exists()`, and when the file is missing it falls back to a built-in
**12-word list** with no warning or error. A typo in `vocab:` would train on 12 words plus pseudo-words, and
the run would look normal. Verified by running it: "vocabulary words loaded: 12". Section B of the script
prints the loaded word count and warns. Severity: high, because it is silent. Proposed fix (**not
implemented**): raise or log loudly when `vocab_json` is given but not loadable.

**I3 — `max_batch_columns` is not enforced for synthetic samples.** §3.1. Severity: high, because the
baseline's memory safety depends on that cap and the failure mode is the slow-then-spill behaviour already
diagnosed earlier in this project. Proposed fix (**not implemented**), after section C gives the real width
distribution:

```python
# src/data/datasets.py, WidthBucketSampler._widths(): replace the median substitution
med = int(np.median(known)) if known else 256
# with a configurable pessimistic estimate for unknown-width sources, e.g.
unknown = int(self.unknown_width or (np.percentile(known, 50) if known else 256))
```
plus a `unknown_width_px` config key read in `make_sampler`. A zero-code alternative is to lower
`max_batch_columns` for the experiment to `64 × p99(synthetic width)` — but that also changes the baseline's
setting, so the treatment would no longer be an isolated change; the config key is cleaner.

**I4 — Validation-image seed leak.** §3.2: 4 collisions, 3 identical images (0.05%). Proposed one-line fix
(**not implemented**):

```python
# src/data/datasets.py, SynthSource.get()
seed = self.seed_base + idx if self.fixed else (hash((self.seed_base, epoch, idx)) & 0x7FFFFFFF)
# becomes
seed = self.seed_base + idx if self.fixed else ((hash((self.seed_base, epoch, idx)) & 0x7FFFFFFF)
                                                % TRAIN_SEED_LIMIT)   # 800_000_000; frozen val seeds are >= 9e8
```
This cannot affect the frozen baseline, which had no synthetic source, and it leaves 8×10⁸ distinct seeds.

**I5 — Synthetic sources and DataLoader workers have never run together; a fork deadlock exists on Linux.**
Tested here: building a synthetic source, **rendering in the parent process**, and then starting DataLoader
workers deadlocks under the `fork` start method (FreeType/HarfBuzz state is not fork-safe). Rendering only
inside workers is fine. Windows uses `spawn`, and a `SynthSource` pickles (42 KB) and generates correctly
after a round trip, so Windows training is expected to be unaffected — **but this has never been run**.
Section E is the proof, and it must pass before training. `tools/bench_synth.py` therefore runs section E
before any rendering happens in its own process. Severity: medium for this project, high for portability.

**I6 — The synthetic training path is still untested end to end.** Consistent with the design review. Section
E exercises exactly the trainer's code (`build_source`, `make_loader`, `TrainStep`) on a mixed source list.

**I7 — 3.7% of synthetic training draws repeat a seed** (111,840 of 3,029,020), so those images appear more
than once across the run. This is inherent to drawing `idx` uniformly from a 1,000,000-item virtual source and
is not a defect; it is recorded because "on-the-fly synthetic data" might otherwise be assumed to be all
distinct.

---

## 6. Estimated training cost

**Measured (existing, from the frozen record and earlier benchmarks):**

- Baseline: 40 epochs × 75,736 samples plus validation in 5 h 45 min (about 8.6 min per epoch overall).
- `bench2_out.txt`: 236 img/s with width bucketing (28% padding); 57 img/s with random batches (50% padding);
  validation of 11,598 words in about 0.8–1.0 min.

**Not measured:** synthetic generation throughput on this machine, mixed-loop throughput, mixed-loop GPU
memory, and bucketing efficiency with synthetic samples present. No figure for these is invented here.

**Estimates, all conditional on the above:**

| Scenario | Basis | Estimate |
|---|---|---|
| Lower bound | Synthetic samples cost the same as handwritten; 2× samples per epoch; plus synthetic monitoring validation (6,000 images/epoch ≈ 0.5 min × 40) | ≈ 11.8 h |
| Bucketing-degraded | Synthetic half effectively unbucketed (median-width estimate), at the measured random-batch rate of 57 img/s | ≈ 20–21 h |
| Generation-bound | CPU generation cannot keep up with the GPU | not bounded by current evidence |

Section G of the script replaces this range with a measured figure, from which `max_hours` should be set with
headroom. Until then, `max_hours` for the treatment cannot be chosen responsibly.

---

## 7. Regression margin (prerequisite 7)

**Pre-registered design parameter, recorded unchanged and not altered here:**

> δ_CER = **+0.10 percentage points**, δ_WER = **+0.50 percentage points**, absolute, on the IIIT test set,
> treatment minus control.

This is a **declared tolerance, not a statistically validated noise floor.** The baseline was trained with one
seed, so its run-to-run variance is unknown. The only related measurements are the baseline's
validation-versus-test differences, 0.028 pp CER and 0.046 pp WER, which are not seed-noise measurements.
Any change to δ must be explicit and recorded before the treatment's test-set evaluation.

---

## 8. Exact changes required before training

None of these has been made.

| # | Change | Kind | Needs approval |
|---|---|---|---|
| C1 | Create `configs/exp001_mixed_synth.yaml`: a copy of `configs/baseline_hw.yaml` with a new `experiment:` name, `iiit_hw` at weight 0.5, a synthetic source with an explicit `vocab:` path, the chosen style arrangement, `epoch_size: 151472`, the synthetic monitoring validation entries at `primary: false`, and `max_hours` from the benchmark | new config file | yes — explicitly deferred by the current instruction |
| C2 | Seed-range guard in `SynthSource.get` (I4) | `src/data/datasets.py`, one line | yes — or choose option 2/3 in §3.2 |
| C3 | Unknown-width estimate for bucketing (I3), after section C measures the distribution | `src/data/datasets.py` + one config key in `src/train.py` | yes — or accept a measured-safe `max_batch_columns` |
| C4 | Loud failure when a given `vocab:` cannot be loaded (I2) | `src/synth/text_sampler.py` | recommended |
| C5 | `--style` filter and per-sample output in `tools/evaluate.py` (design review §6.3) | `tools/evaluate.py` only | yes, before the evaluation stage |
| C6 | Run `tools/bench_synth.py` and record `bench_synth_out.txt` | no code change | no — read-only |

---

## 9. Remaining decisions requiring explicit approval

1. **Seed leak:** apply the one-line guard (C2), or accept and document the 3 identical images.
2. **Bucketing safety:** config key for the unknown-width estimate (C3), or a measured-safe
   `max_batch_columns` for the experiment, accepting that this differs from the baseline value.
3. **Style arrangement:** one synthetic source at `old_print_prob: 0.5` (binomial split, matches design review
   §6.2) or two sources at 0.0 and 1.0 (exact split, separate logging).
4. **Epoch size:** 151,472 preserving handwriting exposure while doubling steps, or 75,736 preserving steps
   while halving handwriting exposure. Neither preserves both; the design review recommends the former.
5. **`max_hours`:** to be set from the benchmark, not before it.
6. **δ:** confirm +0.10 pp CER / +0.50 pp WER as pre-registered, or change it now and record the change.
7. **Whether C4 and C5 are made before the treatment run** or only before evaluation.

---

## 10. Integrity statement

- Created: this report and `tools/bench_synth.py` (read-only audit; writes no file, trains nothing).
- Not modified: `runs/baseline_hw/`, `configs/baseline_hw.yaml`, `configs/mixed_print_hw.yaml`,
  `data/synth_val/`, `data/real/`, `src/`, all existing `reports/` outputs.
- No experiment configuration created. No training started. No checkpoint written. No dataset regenerated.

---

BASELINE REMAINS FROZEN
E1 COMPLETE
E4 COMPLETE
STYLE DIAGNOSTIC COMPLETE
EXPERIMENT_001 DESIGN COMPLETE
PREREQUISITES CHECKED
NO TRAINING STARTED
NO EXPERIMENT CHECKPOINT CREATED
