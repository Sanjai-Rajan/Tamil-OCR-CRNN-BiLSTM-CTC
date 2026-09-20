# Final Project Skeleton Inventory

## 1. Project Root
`C:\Users\prsan\Desktop\Sem 5\C-DAC Projects scraps\C-DAC\1_Draft`

## 2. Complete Structural Tree
- `app/`: Production Web Application and OCR API.
- `checkpoints/`: Trained OCR (CRNN) Checkpoints (127 GB).
- `data/`: Datasets, Text Corpora, and Generated Vocabularies (56 GB).
- `docs/`: Documentation, Audits, and Evaluation Reports.
- `models/`: Neural Network Architectures and Computer Vision Scripts (523 MB).
- `outputs/`: Training Logs, Histories, and Experimental Artifacts (1.4 GB).
- `scripts/`: Training, Evaluation, and Diagnostic utility scripts.
- `venv/`: Python virtual environment.

## 3. Production Pipeline
The active OCR system serving the frontend uses the following path mapping:
1. **Frontend**: `app/templates/index.html` → `app/static/app.js`
2. **Backend**: `app/main.py`
3. **Inference Pipeline**: `app/inference.py`
4. **Segmentation**: `models/cv/segmenter.py` (Classical CV)
5. **OCR Recognition**: `models/recognition/crnn.py` (CRNN Architecture + CTC Decoding)
6. **Deterministic Correction**: `models/digitalization/conceptual_corrector.py` utilizing `data/tamil_ocr_dataset/vocabulary/tamil_correction_vocabulary.json`

## 4. Training Code
The primary code used to train the system from scratch:
- `scripts/train_crnn.py`: Orchestrates loading, augmentation, and training of the CRNN using CTC Loss.
- `scripts/dataset_loader.py`: Handles dataset indexing and parsing.
- `models/recognition/encoder.py`: Base CNN initialization.
- `models/digitalization/tokenizer.py`: Handles token-to-index mapping.

## 5. Training Evidence To Preserve
The following paths serve as absolute evidence of local development and model training and **MUST NOT BE DELETED**:
- `scripts/train_crnn.py`
- `scripts/dataset_loader.py`
- `data/tamil_ocr_dataset/` (The raw training data itself)
- `checkpoints/recognition/tamil/` (Iterative, intermediate `.pth` files proving training evolution)
- `outputs/training/tamil/` (CSV histories, loss curves, and JSON state files proving progressive learning)
- `models/recognition/crnn.py` and its components

## 6. CRNN Checkpoints To Preserve
All CRNN checkpoints located under `checkpoints/recognition/tamil/`. These files average ~130MB in size and contain optimizer and scheduler state for resumption, providing undeniable proof of training. Key locations:
- `checkpoints/recognition/tamil/tamil_with_tamilnet/` (`best.pth`, `latest.pth`, `epoch_*.pth`)
- `checkpoints/recognition/tamil/tamil_with_tamilnet_smoke/`
- `checkpoints/recognition/tamil/tamil_with_tamilnet_test/`
- `checkpoints/recognition/tamil/experiments/gpu_lr_3e-4_bs32/`
- `checkpoints/recognition/tamil/experiments/lr_1e-4_bs32/`
- `checkpoints/recognition/tamil/experiments/tiny_overfit_500/`

## 7. Training Histories To Preserve
Loss curves, mode-collapse metrics, and evaluation tracking located at:
- `outputs/training/tamil/tamil_with_tamilnet/training_history.csv`
- `outputs/training/tamil/tamil_with_tamilnet/training_history.json`
- `outputs/training/tamil/tamil_with_tamilnet/training_state.json`
*(And corresponding histories in all the `experiments/` subdirectories).*

## 8. Dataset Inventory
- `data/tamil_ocr_dataset/`: Primary training images and labels.
- `data/TamilNet_old/`: Early/legacy dataset files.
- `data/text_corpus/`: The Project Madurai corpus used for vocabulary building.
- `data/tamil_ocr_dataset/vocabulary/`: The extracted deterministic dictionaries.
*(Total Size: ~56.4 GB, ~799,715 files).*

## 9. Experimental / Pretrained AI Artifacts
These files exist in the project structure but are **NOT** utilized by the production OCR pipeline (as verified by the prior forensic audit). 
- `models/indic_bert/pytorch_model.bin` (134 MB)
- `outputs/mlm_poc_4060/checkpoints/best_model/model.safetensors` (132 MB)
- `outputs/pre_5090_final/temp_ckpt/model.safetensors` (1.2 GB)
- `scripts/validate_ocr_restoration.py`
- `scripts/train_mlm_4060.py`
- `scripts/benchmark_mlm_5090.py`

## 10. Frontend / Backend
- **Backend API**: `app/main.py`, `app/inference.py`
- **Active Frontend Web App**: `app/templates/index.html`, `app/static/app.js`, `app/static/style.css`
- *(Legacy/Prototype UI)*: `demo/` folder (Streamlit-based or similar, currently inactive in production).

## 11. Evaluation and Diagnostics
Files utilized to test, debug, and diagnose OCR performance without modifying the core pipeline:
- `scripts/test_frontend.py`
- `scripts/debug_multiline.py`
- `scripts/eval_inference_ocr_temporal_fix.py`
- `scratch_diagnostic_fast.py`

## 12. Documentation
Foundational reporting and auditing evidence stored in `docs/`:
- `docs/OCR_TEMPORAL_FIX_TEST_REPORT.md`
- `docs/CONCEPTUAL_CORRECTION_IMPLEMENTATION.md`
- `docs/FINAL_AI_PRETRAINED_API_AUDIT.md`

## 13. Dependencies
- **Deep Learning**: `torch`, `torchvision`, `torchaudio`
- **Classical CV**: `opencv-python`, `pillow`, `albumentations`
- **Web**: `fastapi`, `uvicorn`, `streamlit`
- **Utils**: `numpy`, `pandas`, `scikit-learn`

## 14. Git / Version Control
- **Git Status**: `.git` directory is NOT present. The project is not currently tracked by a local git repository. All version history is implicit in the `checkpoints` and `outputs/training` folders.

## 15. Size Summary
- **checkpoints/**: ~127.4 GB (1,032 files)
- **data/**: ~56.4 GB (799,715 files)
- **venv/**: ~5.2 GB (38,040 files)
- **outputs/**: ~1.4 GB (1,014 files)
- **models/**: ~523 MB (71 files)
- **scripts/**: ~0.64 MB (102 files)
- **app/**: ~0.06 MB (22 files)
- **docs/**: ~0.15 MB (36 files)

## 16. Cleanup Candidates

### MUST PRESERVE
- **All of `checkpoints/recognition/tamil/`** (Do NOT delete, this is the training evidence).
- **All of `outputs/training/`** (Do NOT delete, contains CSV loss histories).
- **All of `data/`** (Do NOT delete, this is the source dataset).
- **All of `app/`** (Production code).
- **All of `models/recognition/` and `models/cv/`**.
- **All of `scripts/`**.
- **All of `docs/`**.

### SAFE TO CONSIDER REMOVAL
*(These are strictly candidates. Nothing has been deleted yet. Deleting these removes experimental NLP/LLM traces without harming the CRNN OCR evidence).*
- `models/indic_bert/pytorch_model.bin`
- `outputs/mlm_poc_4060/`
- `outputs/pre_5090_final/`
- `scripts/train_mlm_4060.py`
- `scripts/validate_ocr_restoration.py`
- `scripts/benchmark_mlm_5090.py`
- Legacy folders like `baseline_before_character_stage/` (0.44 MB)
- `.pytest_cache/`
- `scratch/`
- `venv/` (Can be regenerated from requirements.txt to save 5GB)
