# Experiment 002-A Final Evaluation Report

## 1. Experiment Identity
Experiment 002-A
Hypothesis H-002: Building every training batch from both sources in their configured 50/50 proportion (paired batches)... reduces the IIIT handwriting regression observed in Experiment 001 while retaining Experiment 001's synthetic-print improvement.

## 2. Frozen Artifact Hashes
- `runs/baseline_hw/best.pth`: 0b2655176054793ce6cfc7a41f7a90eefd8f19320c30fa3b6a7b302127b5fe1b
- `runs/experiment_001/best.pth`: d095b2e84daea177bb35d5ae386e1670e6ca3c27af685d9491f5ced08e086f2b
- `runs/experiment_002a/best.pth`: fb29131ed71fe0c89229497d0171b1ff2de4e1e581b4d8c1e09d6fbbaf62d5f1
- `configs/baseline_hw.yaml`: 11f4570724411bbb770f96856bbe2754375dddb52a4e513cf3f6f5b8fdc77724
- `configs/experiment_001.yaml`: 311d2d142ea45db39cfe8d5f1120dfb098bdaddd8207e0c4f1f41a8d750b3b3d
- `configs/experiment_002a.yaml`: 4588dff5ab90aecada8aa4fe4b8bc723832be11d036163788e8a5ed8af3f0f32
- `data/real/iiit_hw/manifest_test.jsonl`: 907cc14448fac442bb4ccfd1656f51dbbaff78b3b36df6caffad7b8f599292b3
- `data/synth_val/manifest.jsonl`: e9203dff3b9e7ea6adbdf445209263b7672fc23cebaf86f5536b14dc578e575c

## 3. Evaluation Datasets
- IIIT Test Set (`data/real/iiit_hw/manifest_test.jsonl`): 16,184 samples
- Synthetic Validation Set (`data/synth_val/manifest.jsonl`): 6,000 samples (3,000 modern, 3,000 old)

## 4. Evaluation Methodology
- Model evaluation: `tools/evaluate.py --samples`
- Paired comparison: `tools/compare_eval.py`
- Paired bootstrap: 10,000 resamples, seed 20260923, two-sided 95% CI.
- CER recomputed from totals in each resample.
- Exact commands followed preregistration.

## 5. IIIT Test Results (All 16,184 samples)
| Model | CER | WER |
|---|---|---|
| Frozen Baseline | 0.6006% | 4.4550% |
| Frozen Exp 001 | 0.7524% | 5.5425% |
| Experiment 002-A | 0.6113% | 4.5230% |

## 6. Synthetic Results
| Set | N | CER | WER |
|---|---|---|---|
| Combined | 6000 | 0.1671% | 1.1000% |
| Modern (All) | 3000 | 0.0618% | 0.4333% |
| Old (All) | 3000 | 0.2724% | 1.7667% |

## 7. 002-A vs 001 Paired Comparison (Primary)
- **Primary Metric (IIIT CER):** Δ -0.1410 pp, 95% CI [-0.1752, -0.1075]
- **WER:** Δ -1.0195 pp, 95% CI [-1.2790, -0.7600]

## 8. 002-A vs Baseline Safety Comparison
- **IIIT CER:** Δ +0.0107 pp, 95% CI [-0.0207, +0.0427] 
  - Threshold (<= +0.10 pp): **MET**
- **IIIT WER:** Δ +0.0680 pp, 95% CI [-0.1792, +0.3089]
  - Threshold (<= +0.50 pp): **MET**

## 9. Synthetic Retention Comparison (002-A vs Baseline)
- **Covered Modern CER:** Δ -15.7518 pp, 95% CI [-16.9556, -14.6034]
- **Covered Old CER:** Δ -17.7971 pp, 95% CI [-18.8565, -16.7535]

## 10. Confidence Intervals & 11. McNemar Result
- All CIs reported above used 10,000 resamples with numpy seed 20260923.
- McNemar tests:
  - 002-A vs 001 IIIT test: exact p=2.2542e-14
  - 002-A vs Baseline IIIT test: exact p=6.2272e-01

## 12. H-002 Preregistered Decision
**H-002 SUPPORTED**
- Primary criterion (CI upper bound < 0): MET
- Synthetic retention criterion: MET

## 13. Batch composition / throughput observations
- Training throughput ~ 190 img/s
- Batch size reached median 44
- Realised composition: single_source_batches_pct 0.4%, mean_batch_size 43.9

## 14. Limitations
- Single-seed run. Seed-to-seed variance is not covered by the CIs.
- Batch size, step count, and padding share differences co-occur with the pairing manipulation, preventing isolated causal attribution to batch grouping alone.

## 15. Exact Commands Used
- Evaluated via `tools/evaluate.py --samples` with `--ckpt` for the three checkpoints on `manifest_test.jsonl` and `data/synth_val/manifest.jsonl` (with `--style modern`/`old`).
- Compared via `tools/compare_eval.py --iiit-test <c> <t> --synth-combined <c> <t> --synth-modern <c> <t> --synth-old <c> <t> --train-manifest data/real/iiit_hw/manifest_train.jsonl --out <out>`.

## 16. Artifact Integrity Statement
- Confirmed no baseline checkpoint changed.
- Confirmed Experiment 001 checkpoint unchanged.
- Confirmed Experiment 002-A checkpoint unchanged.
- Confirmed configs unchanged.
- Confirmed frozen synthetic images unchanged.
- Confirmed manifests unchanged.

---

### Compare against all three important references
| Model | IIIT Test CER | IIIT Test WER | Synth Combined CER | Synth Combined WER | Modern CER | Old CER |
|---|---|---|---|---|---|---|
| Baseline | 0.60% | 4.46% | 19.16% | 47.35% | 18.51% | 19.82% |
| Experiment 001 | 0.75% | 5.54% | 0.26% | 1.78% | 0.11% | 0.42% |
| Experiment 002-A | 0.61% | 4.52% | 0.17% | 1.10% | 0.06% | 0.27% |
