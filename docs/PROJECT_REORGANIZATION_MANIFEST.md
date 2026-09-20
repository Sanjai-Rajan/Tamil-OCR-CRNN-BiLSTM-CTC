# Project Reorganization Manifest

## 1. Root-Level Files
The following files are currently loose in `1_Draft/` and will be moved to maintain a clean root.

| File | Type | Category | Proposed Destination | Reason | Risk |
|---|---|---|---|---|---|
| `requirements.txt` | txt | production | (remain in root) | Core dependency file | None |
| `BEST_BASELINE_TRAINING_HISTORY.csv` | csv | training evidence | `outputs/training/full_tamil_baseline/` | Associates history with baseline | None |
| `BEST_BASELINE_TRAINING_HISTORY.json` | json | training evidence | `outputs/training/full_tamil_baseline/` | Associates state with baseline | None |
| `BEST_40_EPOCH_RESULTS.txt` | txt | training evidence | `outputs/training/global_shuffle_40epoch/` | Experiment output | None |
| `BEST_RUN_ANALYSIS.txt` | txt | report | `docs/training/` | Historical analysis | None |
| `AUDIT_REPORT.md`, `PROJECT_STATUS.md`, `REFINEMENT_BASELINE.md` | md | report | `docs/audits/` | Project tracking | None |
| `CTC_IMPLEMENTATION_EVIDENCE.txt`, `PPT_EVIDENCE.txt` | txt | report | `docs/architecture/` | Implementation notes | None |
| `current_files.txt`, `ckpt_keys.txt` | txt | temporary | `outputs/archive/` | CLI pipeline dumps | None |
| `diagnostic_detailed.json`, `scratch_diag_out.json` | json | diagnostics | `outputs/archive/diagnostics/` | Previous debug runs | None |
| `forensic_analysis.py`, `scratch_diagnose.py`, `temporal_analysis.py`, `tiny_overfit.py` | python | diagnostics/historical | `scripts/historical/` | Root clutter | `sys.path` update needed |
| `sample.jpg`, `sample_10lines.png`, `synthetic_test_line.png` | img | sample/input | `data/samples/` | Test images | None |

## 2. Scripts
The `scripts/` directory will be logically partitioned into subdirectories.

**Move to `scripts/training/`:**
- `train_crnn.py`, `train_overfit_tiny.py`, `run_a100_experiment.py`, `train_character_classifier.py`, `train_refinement.py`
*(Risk: Caller scripts/CLI commands will need to invoke `python scripts/training/train_crnn.py`. Internal imports from `scripts.dataset_loader` will need fixing since `train_crnn.py` will no longer be adjacent to it).*

**Move to `scripts/evaluation/`:**
- `eval_crnn.py`, `evaluate_crnn.py`, `evaluate_ocr.py`, `evaluate_exp_A.py`, `eval_test_balanced.py`

**Move to `scripts/diagnostics/`:**
- `root_cause_diagnostic.py`, `geometry_analysis.py`, `diagnose_ctc.py`, `validate_ctc_length.py`, `scale_diagnostic.py`

**Move to `scripts/data/`:**
- `dataset_loader.py`, `prepare_tamil_ocr_dataset.py`, `build_tamil_vocabulary.py`, `import_tamil_packet.py`, `ds3_audit.py`, `ds6_audit.py`, `split_dataset.py`
*(Risk: `dataset_loader.py` is widely imported as `from scripts.dataset_loader import...`. Moving it will break these imports across all training/diagnostic scripts unless a refactor occurs).*

**Move to `scripts/historical/`:**
- `mlm_smoke_test.py`, `monitor_mlm_live.py`, `mlm_checkpoint_test.py` (Obsolete traces).

## 3. Scratch
`scratch/root_cause_diagnostic.py`
- **Action**: Move to `scripts/diagnostics/root_cause_diagnostic.py`
- **Reason**: It is an active CRNN mode-collapse diagnostic tool, not temporary garbage.
- **Risk**: The script manually sets `sys.path.append(...)`. That logic must be adjusted for its new depth level relative to the root.

## 4. Outputs
`outputs/training/` will remain completely **untouched**.
Other folders (e.g., `architecture_audit/`, `diagnostics/`, `dataset_source_audit/`, `fast_track/`, `path_migration/`, `mlm_5090_audit/`) will be moved into:
`outputs/archive/<folder_name>/`
- **Reason**: Reduces visual clutter while preserving exact historical audits and old ML output evidence.

## 5. Documentation
`docs/` will be organized conceptually:
- `docs/architecture/` (For `CONCEPTUAL_CORRECTION_IMPLEMENTATION.md`, `CURRENT_CRNN_ARCHITECTURE.txt`)
- `docs/audits/` (For `FINAL_AI_PRETRAINED_API_AUDIT.md`, `CLEANUP_EXECUTION_REPORT.md`)
- `docs/evaluation/` (For `OCR_TEMPORAL_FIX_TEST_REPORT.md`)
- **Reason**: Traceability and cleaner structure. No documents will be deleted.

## 6. Samples
All loose `*.png`, `*.jpg` at the root will be moved to `data/samples/`.
- **Reason**: Isolates static test inputs from code files.

## 7. Configuration
`configs/architecture/final_system.yaml` contains references to the now-deleted `models/indic_bert` and `models/mt5_small_restoration/base`.
- **Action**: Leave the file exactly as is, but document it as historical.
- **Reason**: The active OCR system uses `config/model_config.yaml` and does not import `final_system.yaml`. Modifying it rewrites history unnecessarily.

## 8. Requirements
`requirements.txt` was audited:
- **Required**: `torch`, `torchvision`, `torchaudio`, `opencv-python`, `pillow`, `fastapi`, `uvicorn`, `albumentations`, `numpy`, `pandas`.
- **Absent**: `transformers` and `huggingface_hub` are **NOT** listed in `requirements.txt`.
- **Action**: No changes needed. The file is perfectly clean.

## 9. Virtual Environment
The `venv/` is ~5.2 GB and still has `transformers` and `huggingface_hub` installed (leftover from manual `pip install` during NLP experiments).
- **Status**: The production pipeline `app/inference.py` only imports PyTorch, PIL, torchvision, and local code.
- **Action**: The entire `venv/` can be safely deleted and instantly rebuilt from the clean `requirements.txt`.

## 10. Proposed Final Tree
```text
1_Draft/
├── app/
├── checkpoints/
├── config/               # Active config
├── configs/              # Historical configs
├── data/
│   ├── tamil_ocr_dataset/
│   ├── TamilNet_old/
│   ├── text_corpus/
│   └── samples/          # Moved root images
├── docs/
│   ├── architecture/
│   ├── audits/
│   └── evaluation/
├── models/
├── outputs/
│   ├── training/         # PRESERVED
│   └── archive/          # Moved historical audits
├── scripts/
│   ├── training/
│   ├── evaluation/
│   ├── diagnostics/
│   ├── data/
│   └── historical/
└── requirements.txt
```

## 11. Proposed Moves
- **~65** loose root-level files moving into `docs/`, `outputs/archive/`, `scripts/historical/`, or `data/samples/`.
- **~90** Python scripts moving into categorized subdirectories in `scripts/`.
- **~19** `outputs/` subdirectories moving into `outputs/archive/`.
- **1** file moving from `scratch/` to `scripts/diagnostics/`.

## 12. Proposed Deletions
- `venv/` (To trigger a clean 1.5GB PyTorch rebuild without the 4GB NLP bloat).
- `.pytest_cache/`
- Empty `scratch/` directory.

## 13. MUST PRESERVE
The following paths are explicitly shielded from modification:
- `checkpoints/recognition/tamil/` (Including `ocr_balanced_temporal/best.pth`)
- `outputs/training/`
- `data/tamil_ocr_dataset/`
- `data/TamilNet_old/`
- `data/text_corpus/`
- `models/recognition/`
- `models/cv/`
- `models/digitalization/`
- `app/`
- `scripts/train_crnn.py` (File contents preserved)
- `scripts/dataset_loader.py` (File contents preserved)

## 14. Dependency Risks
**HIGH RISK: Python Import Paths**
Moving `scripts/dataset_loader.py` to `scripts/data/dataset_loader.py` breaks all `from scripts.dataset_loader import ...` calls in training, evaluation, and diagnostic scripts.
- **Mitigation Strategy**: The refactoring phase must run an automated search-and-replace (e.g., changing `from scripts.dataset_loader` to `from scripts.data.dataset_loader`) across all moved files. Additionally, any script utilizing `sys.path.append(...)` must adjust the relative depth (`parent.parent` vs `parent`).

## 15. Execution Order
1. Freeze current checkpoint (`ocr_balanced_temporal`).
2. Freeze all training evidence folders.
3. Validate `requirements.txt` (Confirmed clean).
4. Delete `venv/` and execute `pip install -r requirements.txt`.
5. Create new folder structure (`scripts/training`, `outputs/archive`, etc.).
6. Reorganize loose outputs and documentation.
7. Reorganize root files and sample images.
8. Reorganize `scripts/` and perform global `import` path replacements.
9. Move `scratch/root_cause_diagnostic.py` and delete `scratch/`.
10. Run `app/main.py` test to verify the new structure boots cleanly.
