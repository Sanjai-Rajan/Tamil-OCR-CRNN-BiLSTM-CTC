# Synthetic Style Diagnostic — Planning Report (Modern vs. Old-Print Split)

Read-only diagnostic preparation. No training was run, no baseline files were touched, no synthetic images were regenerated, and no code changes were implemented — this document is a plan only.

---

## 1. Where the style information is stored

Every entry in `data/synth_val/manifest.jsonl` already carries a `"style"` field, e.g.:

```json
{"image": "images/modern_000000.png", "label": "தூறிய", "style": "modern", "font": "arima"}
```

This field is **not inferred from image appearance** — it is written directly at generation time by `tools/synth_export.py` (lines 35–42), which loops over exactly two style batches:

```python
for style, n, p_old, offset in (("modern", a.n_modern, 0.0, 0), ("old", a.n_old, 1.0, 50_000_000)):
    gen = SynthWordGenerator(a.fonts, vocab_json=a.vocab, old_print_prob=p_old, text_seed=a.seed_base + offset)
    for i in range(n):
        img, text, meta = gen.sample(a.seed_base + offset + i)
        name = f"{style}_{i:06d}.png"
        ...
        mf.write(json.dumps({"image": f"images/{name}", "label": text, "style": style, "font": meta["font"]}, ...) + "\n")
```

The `style` value written to the manifest is the exact loop variable that controlled `old_print_prob` (0.0 for the "modern" batch, 1.0 for the "old" batch) at the moment each image was generated — i.e., it is ground truth from the generator's own configuration, not a label assigned after the fact. The image filename (`modern_NNNNNN.png` / `old_NNNNNN.png`) is written from the same `style` variable in the same loop iteration, so filename and manifest field are two redundant recordings of the same generation-time fact, not two independent guesses.

`src/data/datasets.py:41-56` (`ManifestDataset`) already supports filtering by exactly this field, with this exact use case documented in its own docstring:

```python
def __init__(self, manifest: str, name: str | None = None, augment=None, max_items: int | None = None,
             seed: int = 0, where: dict | None = None):
    """where: keep only entries whose fields match, e.g. {"style": "old"} for the synthetic val set."""
    ...
    if where and any(e.get(k) != v for k, v in where.items()):
        continue
```

This `where` filtering mechanism is already used elsewhere in the codebase (`src/train.py`'s `build_source`/`build_val` pass through `spec.get("where")`) — it is existing, exercised infrastructure, not something that needs to be invented for this diagnostic.

## 2. Counts

Verified by reading and parsing all 6,000 lines of `data/synth_val/manifest.jsonl` directly (read-only):

| Style | Count |
|---|---|
| modern | 3,000 |
| old | 3,000 |
| **Total** | **6,000** |

## 3. Can every one of the 6,000 samples be assigned unambiguously?

Yes. Verified directly:
- `style` field present and non-empty on all 6,000 entries — **0 missing**.
- Filename prefix (`modern_`/`old_`) matches the entry's `style` field on all 6,000 entries — **0 mismatches**.
- Exactly two distinct style values appear (`modern`, `old`) — no ambiguous, blank, or third-category values.

There is no guessing involved and no reliance on image appearance (per requirement 4) — the assignment is a direct, exact-match read of an existing, generation-time-authoritative field.

## 4. Exact files that would need to be touched to implement the diagnostic

**Only one file: `tools/evaluate.py`.**

`src/data/datasets.py` needs **no change** — the `where` filter parameter it needs already exists and is already documented for this exact purpose (see §1).

The smallest possible change to `tools/evaluate.py` is to add one optional CLI argument and thread it into the existing `ManifestDataset(...)` call:

```python
ap.add_argument("--style", choices=["modern", "old"], default=None,
                help="filter data/synth_val/manifest.jsonl (or any manifest with a 'style' field) to one style")
...
ds = ManifestDataset(a.manifest, where={"style": a.style} if a.style else None)
```

This is a ~2-line addition (one `add_argument` call, one changed constructor call) to a single file. No change to `src/data/datasets.py`, `src/train.py`, `configs/baseline_hw.yaml`, or any manifest/data file is required. Running the same command three times (once with `--style modern`, once with `--style old`, once with no `--style` flag for the existing combined result) would produce the three-way modern/old/combined breakdown requested.

**This change has not been implemented** — per your instruction, it is proposed only.

## 5. Can this diagnostic be performed without changing the baseline?

Yes, fully. The proposed change:
- Only affects `tools/evaluate.py`, a standalone read-only scoring tool — it is not imported by `src/train.py` and has no effect on training.
- Does not touch `runs/baseline_hw/` (checkpoints, logs, history) in any way — evaluation only reads `best.pth`, never writes to it.
- Does not touch `configs/baseline_hw.yaml`.
- Does not touch or regenerate any of the 6,000 synthetic images or `data/synth_val/manifest.jsonl` — it only reads and filters that existing, frozen manifest in memory.
- Requires no retraining, no new experiment run, no external data/API/pretrained model.

The frozen baseline (`runs/baseline_hw/best.pth`, `last.pth`, `configs/baseline_hw.yaml`) remains completely untouched by this diagnostic, both as proposed and if later implemented and run.

---

## Summary

Reliable, generation-time-authoritative style metadata already exists for all 6,000 synthetic validation samples (3,000 modern / 3,000 old, unambiguous, 0 missing, 0 filename/field mismatches), sourced from `tools/synth_export.py`'s own generation loop and exposed via `ManifestDataset`'s existing `where` filter. The diagnostic (splitting the E4 CER/WER numbers by style) requires touching exactly one file, `tools/evaluate.py`, with a ~2-line change, and can be performed without any modification to the frozen baseline.

---

BASELINE REMAINS FROZEN
E1 COMPLETE
E4 COMPLETE
STYLE DIAGNOSTIC: NOT EXECUTED
NO NEW TRAINING STARTED
EXPERIMENT_001 NOT STARTED
