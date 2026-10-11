# Experiment 001 Safety Fixes Report

## Overview
Following the prerequisite benchmark, three specific safety risks were identified and patched in the codebase before any further experimentation.

## Implemented Fixes

### A. Synthetic Width Safety
- **Issue**: `SynthSource.est_width()` returned `None`, forcing `WidthBucketSampler` to fall back to the median width of known real handwriting samples. This resulted in synthetic batches wildly exceeding the memory bounds defined by `max_batch_columns` (observed 33,088 cols vs 32,768 cap).
- **Fix**: Added an `unknown_width_estimate` parameter to `WidthBucketSampler`. When populated, it acts as a direct substitute for `None` width estimates instead of calculating the dynamic median.
- **Integration**: Plumbed the new parameter into `src/train.py` via `tc.get("unknown_width_estimate")`, ensuring that it is fully configurable by experiment configurations and safely defaults to `None` (preserving old behavior) for the baseline config.

### B. Seed Leakage Guard
- **Issue**: Found 3 exact collisions where the synthetic generator produced identical images across the validation and training sets due to seed overlap.
- **Fix**: Introduced a deterministic bitwise-XOR operation (`seed ^= 0x12345678`) in `SynthSource.get()` that activates exclusively when a dynamically generated synthetic training seed falls within the frozen validation set's reserved ranges (`900000000..900003000` and `950000000..950003000`).
- **Safety**: Preserves identical reproducibility for fixed validation sets while aggressively steering training variables away from the frozen subset.

### C. Vocabulary Failure Safety
- **Issue**: If the provided `vocab_json` path in `TextSampler` was completely invalid, the script would silently swallow the error and fall back to a built-in 12-word debug array.
- **Fix**: Replaced the silent `exists()` check with an explicit `FileNotFoundError`. The 12-word fallback now only executes if no `vocab_json` argument is supplied in the first place.

## Verification
- **Focused Testing**: Ran an internal suite verifying `WidthBucketSampler` logic, seed leakage bit-shifts, and explicit `FileNotFoundError` behavior on invalid paths. All 3 tests passed in 22.782s.
- **Baseline Isolation**: None of the changes affected the syntax or default paths of `configs/baseline_hw.yaml`, `runs/baseline_hw/`, or `data/synth_val/`. They remain perfectly frozen.
