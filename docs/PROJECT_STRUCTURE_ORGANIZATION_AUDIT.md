# Project Structure Organization Audit

## 1. Current Root Structure
The project root `1_Draft/` contains standard component directories (`app/`, `models/`, `scripts/`, `data/`, `checkpoints/`, `outputs/`, `configs/`, `docs/`, `venv/`), but is currently overwhelmed by 65+ loose files, including temporary scripts, logs, JSON outputs, and txt reports.

## 2. Loose Root-Level Files
The root directory is highly cluttered. Key files and their classifications:
- **A. Production-critical**: `requirements.txt`
- **B. Training-related**: `BEST_BASELINE_TRAINING_HISTORY.csv`, `BEST_BASELINE_TRAINING_HISTORY.json`, `training_results_for_report.txt`
- **C. Evaluation/diagnostic**: `diagnostic_detailed.json`, `forensic_output.txt`, `scratch_diag_out.json`, `inference_results.txt`
- **D. Documentation/report**: `AUDIT_REPORT.md`, `BEST_RUN_ANALYSIS.txt`, `CTC_IMPLEMENTATION_EVIDENCE.txt`, `PROJECT_STATUS.md`
- **H. Temporary/debug**: `temp.json`, `temp.txt`, `sample.jpg`, `sample_10lines.png`
- **I. Historical development evidence**: `scratch_diagnose.py`, `forensic_analysis.py`, `temporal_analysis.py`, `trace_pipeline.py`, `tiny_overfit.py`

*Recommendation*: All loose scripts and JSON/TXT files should be categorized and moved into `scripts/`, `outputs/`, or `docs/` to keep the root directory clean.

## 3. Configuration Audit
Inspected `configs/architecture/final_system.yaml`.
- **Obsolete references**: Lines 7-8 point to `models/indic_bert`. Lines 12-13 point to `models/mt5_small_restoration/base`.
- **Production Status**: This config is an architectural proposal/historical artifact. The active production OCR pipeline does NOT import it (it relies on `config/model_config.yaml`).
- **Safety**: It is completely safe to remove these sections in the future without breaking the OCR service.

## 4. Scripts Audit
The `scripts/` directory contains 60+ flat scripts.
- **A. Current training**: `train_crnn.py`, `dataset_loader.py`
- **B. Current evaluation**: `eval_crnn.py`, `eval_inference_ocr_temporal_fix.py`
- **D. Current diagnostics**: `root_cause_diagnostic.py`, `geometry_analysis.py`
- **E. Dataset preparation**: `prepare_tamil_ocr_dataset.py`, `import_tamil_packet.py`
- **F. Historical OCR experiment**: `train_overfit_tiny.py`, `preview_multiscale.py`
- **H. Obsolete/pretrained**: (Already deleted in previous step).

*Recommendation*: These scripts could be logically grouped into `scripts/training/`, `scripts/evaluation/`, `scripts/diagnostics/`, `scripts/data/`, and `scripts/historical/`.

## 5. Scratch Audit
The `scratch/` directory currently contains a single file: `scratch/root_cause_diagnostic.py`.
- **Function**: It is an active, detailed CRNN diagnostic script used to calculate CNN Temporal Lengths (T) vs Target Lengths and measure CTC mode collapse.
- **Classification**: **MOVE** (to `scripts/diagnostics/root_cause_diagnostic.py`). It is a useful tool, not temporary garbage.

## 6. Demo Audit
- **Status**: The `demo/` directory **does not exist** in the current project root. Any UI functionality currently resides inside `app/templates/` and `app/static/`.

## 7. Outputs Audit
The `outputs/` directory is cluttered with historical audit folders.
- **Preserve**: `outputs/training/` (Contains crucial CRNN training histories).
- **Historical/Review**: `outputs/architecture_audit/`, `outputs/dataset_source_audit/`, `outputs/fast_track/`, `outputs/diagnostics/`.
*Recommendation*: `outputs/training/` must remain untouched. All other folders can be moved into a single `outputs/historical_audits/` archive directory.

## 8. Checkpoints Audit
The `checkpoints/recognition/tamil/` directory contains massive evidence of iterative development:
- **Baseline**: `full_tamil_baseline/`, `tamil_with_tamilnet/`
- **Temporal Experiments**: `ocr_temporal_fix/`, `ocr_balanced_temporal/` (Current Best)
- **Smoke/Test**: `5090_A2_bucketing_smoke/`, `quick_1epoch_test/`
- **Other**: `experiments/`, `fast_track_ctc_fix/`

*Recommendation*: **DO NOT TOUCH**. The internal structure of the checkpoints directory perfectly mirrors the development history.

## 9. Documentation Audit
`docs/` contains multiple markdown reports (e.g., `OCR_TEMPORAL_FIX_TEST_REPORT.md`, `CLEANUP_EXECUTION_REPORT.md`).
*Recommendation*: Keep intact. They could be separated into `docs/audits/` and `docs/development/` for better navigation later.

## 10. Data Audit
`data/` correctly segments datasets:
- `data/tamil_ocr_dataset/`: Primary ML dataset.
- `data/text_corpus/`: The Project Madurai corpus.
- `data/TamilNet_old/`: Legacy datasets.
*Recommendation*: Keep exactly as is.

## 11. Virtual Environment Audit
- **Status**: `venv/` is active and massive (~5.2 GB).
- **Dependencies**: It still contains `transformers`, `huggingface_hub`, and related NLP libraries from the experimental phase.
- **Production Status**: The active `OCRService` does NOT import these packages.
*Recommendation*: `venv/` can be safely deleted and rebuilt using only the core ML libraries (`torch`, `torchvision`, `opencv-python`, `fastapi`, etc.) defined in a clean `requirements.txt`.

## 12. Dependency / Reference Audit
- Loose scripts in the root typically import local modules directly (`import app.inference`).
- Moving these scripts into `scripts/...` will require updating their path imports (e.g., adding `sys.path.append(...)`).
- Moving `scratch/root_cause_diagnostic.py` into `scripts/` will align it with the rest of the diagnostic scripts.

## 13. Proposed Clean Structure
```text
1_Draft/
├── app/                  # Unchanged (Production FastAPI & Frontend)
├── checkpoints/          # Unchanged (CRNN Training Weights)
├── configs/              # Cleaned of ML references
├── data/                 # Unchanged (Raw Data)
├── docs/                 # Unchanged (Reports & Audits)
├── models/               # Unchanged (CRNN Architecture, Segmenter)
├── outputs/
│   ├── training/         # Unchanged (Loss CSVs)
│   └── archive/          # (New) Move all other output folders here
├── scripts/
│   ├── training/         # (New) train_crnn.py, etc.
│   ├── evaluation/       # (New) eval_crnn.py, etc.
│   ├── inference/        # (New) test_frontend.py
│   ├── diagnostics/      # (New) root_cause_diagnostic.py, etc.
│   ├── data/             # (New) dataset_loader.py, import_*.py
│   └── historical/       # (New) Root-level loose scripts
└── requirements.txt      # Cleaned
```
*(All 60+ loose root-level txt/json/py files will be sorted into `outputs/archive/`, `docs/`, or `scripts/historical/`)*.

## 14. Files That Must Remain Untouched
- **`app/*`** (Production code)
- **`models/*`** (Architecture)
- **`checkpoints/recognition/tamil/*`** (Weights)
- **`outputs/training/*`** (Loss Logs)
- **`data/*`** (Datasets)

## 15. Files That Could Be Moved
- All 65 loose root-level files (to `scripts/historical/`, `outputs/`, or `docs/`).
- `scratch/root_cause_diagnostic.py` (to `scripts/diagnostics/`).
- `scripts/*.py` (into their respective subfolders).

## 16. Files That Could Be Deleted Later
- **`venv/`**: Highly recommended to delete and rebuild to instantly shed ~5GB of HuggingFace dependencies.
- **`.pytest_cache/`** and any empty folders.

## 17. Files Requiring Manual Review
- `configs/architecture/final_system.yaml`: Needs manual editing to remove lines referencing deleted IndicBERT paths.
- `requirements.txt`: Needs review to ensure it only lists necessary CV/Torch packages.

## 18. Recommended Next Cleanup Step
**Rebuild the Virtual Environment and Organize the Root**.
1. Delete `venv/`.
2. Update `requirements.txt`.
3. Create the `scripts/*` subdirectories and move the loose root files/scripts into them.
