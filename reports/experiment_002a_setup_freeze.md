# Experiment 002-A — Setup freeze

Frozen 2026-09-24 22:59 (+05:30), before any Experiment 002-A training. This record identifies exactly what the 40-epoch
run will use. Any file whose SHA-256 differs from the values below at launch time invalidates this freeze.

Pre-registration: `reports/experiment_002a_preregistration_addendum.md`
(SHA-256 `143ab66aa497ff9125a676513b20ef7e726f93d6259a86bbac2a81550f9d8af4`).

---

## 1. Final configuration

`configs/experiment_002a.yaml`, SHA-256 **`4588dff5ab90aecada8aa4fe4b8bc723832be11d036163788e8a5ed8af3f0f32`**

| Key | Value |
|---|---|
| `experiment` | `experiment_002a` |
| `train.batch_mix` | `paired` |
| `train.unknown_width_estimate` | **736** |
| `train.max_batch_columns` | 32768 |
| `seed` / `train.epochs` / `train.epoch_size` | 42 / 40 / 151472 |
| Sources | `iiit_hw` (weight 0.5, `augment: hw`) + `synth_print` (weight 0.5, `old_print_prob: 0.5`, vocab `data/text_corpus/tamil_correction_vocabulary.json`) |

**Difference from Experiment 001** (`configs/experiment_001.yaml` = `runs/experiment_001/config.yaml`, SHA-256
`311d2d142ea45db39cfe8d5f1120dfb098bdaddd8207e0c4f1f41a8d750b3b3d`): exactly two keys differ, `experiment`
(`experiment_001` → `experiment_002a`) and `train.batch_mix` (absent → `paired`). Model, sources, validation and all
other training settings are identical, verified by parsing both files.

## 2. Gate results (details: addendum §15)

| Gate | Result |
|---|---|
| A–B. CPU-only preflight at 736 | **Passed.** 3,443 epoch-0 batches; 137,813 planned over 40 epochs; batch size median/max 44/44; every index used exactly once; deterministic (plan `96fb00958cba38c7…`); 3,443 mixed batches; 75,753 / 75,719 handwriting/synthetic; 6 of 3,443 actual batches over 32,768 columns (maximum 45,056); no errors. `reports/experiment_002a_preflight_736.txt` |
| C–F. Bounded GPU safety check | **Passed.** 188.2 img/s (≥ 90); peak reserved 6.07 GB (timed loop) and 6.98 GB (after replay) of 8.00 GB; 0 allocator retries; all 30 heaviest batches replayed with exact CPU-preflight columns; the 45,056-column batch completed without OOM; no CUDA/DataLoader/OOM/Python errors; validation 17,598 images in 48.5 s; estimate ≈ 9.5 h. `reports/experiment_002a_gpu_safety.txt` |

Caveat: the safety check does not guarantee full-run memory behaviour. Experiment 001 reached 10.5 GB during full
training after a 5.8–6.6 GB benchmark. The exit code 127 after complete output is recorded as a teardown observation
(addendum §15.4).

## 3. Frozen hashes (SHA-256, complete)

**Configuration**

| File | SHA-256 |
|---|---|
| `configs/experiment_002a.yaml` | `4588dff5ab90aecada8aa4fe4b8bc723832be11d036163788e8a5ed8af3f0f32` |
| `configs/experiment_001.yaml` | `311d2d142ea45db39cfe8d5f1120dfb098bdaddd8207e0c4f1f41a8d750b3b3d` |
| `configs/baseline_hw.yaml` | `11f4570724411bbb770f96856bbe2754375dddb52a4e513cf3f6f5b8fdc77724` |

**Source code**

| File | SHA-256 |
|---|---|
| `src/train.py` | `8ca3ae6bd59d95131b46c30776f4ed62df299befb135ab0bcc11c1d954343fac` |
| `src/data/datasets.py` | `1989b8776eba470d9eafab818f7f3f0e8c37386bff3064fac9ae5bd5e46dc488` |
| `src/model/crnn_v2.py` | `1f2da2c820dd79a0eb4d2b97ee5a8f89d1fa0f22b92973738b9955b364429f9a` |
| `src/augment.py` | `d66093dad1eb3e9b91cc264f33fc8ace061680bd4007ced8ee69549c0fcb2e0e` |
| `src/charset.py` | `1a9ce37124a2894654b190f2fc8f719c3152ab6c0c6e375504de88e4d73dab14` |
| `src/metrics.py` | `880c4a07ae51f48fc43c4b80e8d114f8f7ec247e7a2c3d6537e441f94160aad1` |
| `src/preprocess.py` | `975893f9cf2e0b6f0ac336d8d937113dbe2cc078a0f9849900061439cc415467` |
| `src/synth/generator.py` | `024c04db3a032fc501b6a968b328c44c4a30c131311a85db91739a4f99991b6f` |
| `src/synth/degrade.py` | `26175ff7d1cfde813cecb97d1871eb0796dcf7e60372f5899001c9b097b5c6bc` |
| `src/synth/render.py` | `04b28bc5e9e26cf86f7a96452e50f7a3b5a82224d58420e504043e4929863f5b` |
| `src/synth/fonts.py` | `3a0696313c5391b49866f63a40b3ccd4b8d867424c7fa424b04c753c21d28f8e` |
| `src/synth/text_sampler.py` | `f7f27c0193765adb281d9c946ead8589383314dd1bf1880c18ee9801c182c090` |

**Tools**

| File | SHA-256 |
|---|---|
| `tools/bench_synth.py` (tested version) | `651e1176c3c11952a25f0cbb8dc96a188b1fe7d2a3da990103857d00e350c6ad` |
| `tools/evaluate.py` | `5368d1175ebe8929c27a9ebe4683d077a659e2266a053d4fd7715b73dc83d70c` |
| `tools/compare_eval.py` | `d629b4b4a298dfed1c59b51fc1a6a7f7b3f794f297162ab84bd99a376318bb77` |

**Data manifests and vocabulary**

| File | SHA-256 |
|---|---|
| `data/real/iiit_hw/manifest_train.jsonl` | `f13e55380548ef5d1480d235ca20b8c8a3732c374d5051b30a672b1b92387a0f` |
| `data/real/iiit_hw/manifest_val.jsonl` | `47cbe15b22b1d7dec621882f2a9f185894f3c6fb82b596b5aa87f7046fd36aab` |
| `data/real/iiit_hw/manifest_test.jsonl` | `907cc14448fac442bb4ccfd1656f51dbbaff78b3b36df6caffad7b8f599292b3` |
| `data/synth_val/manifest.jsonl` | `e9203dff3b9e7ea6adbdf445209263b7672fc23cebaf86f5536b14dc578e575c` |
| `data/text_corpus/tamil_correction_vocabulary.json` | `3f19ac2db418d3061162acfd978463b3f3a0c21652e4968c8383fa2e24a2241a` |

**Frozen comparators (untouched)**

| File | SHA-256 |
|---|---|
| `runs/baseline_hw/best.pth` | `0b2655176054793ce6cfc7a41f7a90eefd8f19320c30fa3b6a7b302127b5fe1b` |
| `runs/baseline_hw/last.pth` | `24f62bae9c66b3b0f00e551b8ea21569e4736275dcf0d239a25f9b24ddb8c144` |
| `runs/experiment_001/best.pth` | `d095b2e84daea177bb35d5ae386e1670e6ca3c27af685d9491f5ced08e086f2b` |
| `runs/experiment_001/last.pth` | `4cca9b4229f246695b47f275755a1725939aec6cee2d514c68e24ea1d5267310` |
| `reports/eval_test_baseline_rescore_samples.jsonl` | `466429cf3dcbb3cc12d5c8f0b5666d4fdee46e3f81a0b83b56b079d3a7ef9603` |
| `reports/eval_test_exp001_samples.jsonl` | `db30c21392e398cc251f4475327d43c7e72b835dd284551ae7f338d01dc5835a` |

**Evidence**

| File | SHA-256 |
|---|---|
| `reports/experiment_002a_preflight_736.txt` | `a4af732c079273882c8473cc2bffed1ce72d86955c98c9d9f0e354c5aff576d3` |
| `reports/experiment_002a_gpu_safety.txt` | `1ea61f229ee5cf562792e798b0f9303628a34cd8915b1e1dc9a1f3c44a37d3a6` |
| `<scratchpad>/preflight_002a_out.txt` (517 record, not copied) | `ac7f9ba3cc9248d49ab46c8be88e8e63408c69ab2745d3c2fae39d29b0a3ad90` |

## 4. State at freeze

- Baseline (`runs/baseline_hw/`, `configs/baseline_hw.yaml`) and Experiment 001 (`runs/experiment_001/`,
  `configs/experiment_001.yaml`) artifacts are unchanged: checksums verified identical to the pre-freeze snapshot.
- **No Experiment 002-A training has run and no 002-A checkpoint exists**: `runs/experiment_002a/` does not exist.
  `runs/` contains `baseline_hw`, `exp001_bn_d0`, `experiment_001`.
- Planned launch command (not run): `.venv/Scripts/python.exe -m src.train --config configs/experiment_002a.yaml`
  (fresh initialisation, no `--resume`).
- Remaining before launch: final training approval (addendum §13, item 8). The 517 preflight and no-guard artifacts
  remain in the scratchpad only (addendum §13, item 6).
