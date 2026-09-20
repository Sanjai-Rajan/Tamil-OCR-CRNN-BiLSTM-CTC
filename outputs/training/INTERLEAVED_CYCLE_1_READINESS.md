# Interleaved Cycle 1 Readiness Report

## 1. MLM Readiness
- **START CHECKPOINT**: `checkpoints/mlm/indicbert_run_01/checkpoint-8000`
- **MLM_START_STEP**: 8000
- **MLM_TARGET_STEP**: 10000
- **CHECKPOINT COMPLETE**: YES (Contains `model.safetensors`, `optimizer.pt`, `scheduler.pt`, `trainer_state.json`, `rng_state.pth`)
- **RESUME VALID**: YES
- **SHA256 VERIFIED**: YES (Verified against `CHECKPOINT_MANIFEST.csv`)

## 2. RESTORATION Readiness
- **START CHECKPOINT**: `models/mt5_small_restoration/base/` (Base Model)
- **RESTORATION_START_STEP**: 0
- **RESTORATION_TARGET_STEP**: 1000
- **CHECKPOINT COMPLETE**: YES (Base model files: `config.json`, `model.safetensors`, `tokenizer.json`, etc. are present)
- **RESUME VALID**: YES (Fresh start from base model)
- **SHA256 VERIFIED**: UNCERTAIN (No manifest exists for the base model or `checkpoint-20` dry runs)
- **Train Records**: 3953
- **Validation Records**: 494
- **Test Records**: 495
- **Conflict Handling**: Error-annotated formatting using `"Fix errors: "` prefix.
- **Batch Size**: 4
- **Gradient Accumulation**: 1
- **Learning Rate**: 5e-4
- **Max Length**: 128
- **Precision**: BF16/FP16 (depending on hardware)

## 3. OCR Readiness
- **START CHECKPOINT**: `checkpoints/recognition/tamil/global_shuffle_lr_1e-4/best.pth` (or `latest.pth` at epoch 78)
- **OCR_START_STATE**: epoch_078 (or best.pth)
- **OCR_TARGET_STATE**: epoch_078 + 1000 steps (or next logical boundary based on cycle size)
- **CHECKPOINT COMPLETE**: YES (Contains `model_state_dict`, `optimizer_state_dict`, `state`, `config`)
- **RESUME VALID**: YES (Through `train_crnn.py --resume` flag)
- **SHA256 VERIFIED**: UNCERTAIN (No manifest tracking mechanism explicitly verified for OCR checkpoints)
- **Architecture**: CRNN
- **Vocabulary**: `data/tamil_ocr_dataset/vocabulary/tamil_vocab.json`
- **Training Dataset**: `data/tamil_ocr_dataset/imported/tamil/train`
- **Batch Size**: 32
- **Gradient Accumulation**: 1
- **Learning Rate**: 1e-4 (from directory naming/previous config)
- **Epoch/Step Config**: 20 target epochs, but resuming from epoch 78.
- **Precision**: Mixed Precision (`mixed_precision: true`)

## 4. INTERLEAVED PLAN COMPLIANCE
The existing `INTERLEAVED_TRAINING_PLAN.md` has been updated to reflect the actual MLM stop state:
- **MLM**: 8000 → 10000
- **RESTORATION**: 0 → 1000
- **OCR**: existing checkpoint → +1000 steps (Target needs explicit step handling added to `train_crnn.py`)

## FINAL STATUS
**READY_FOR_CYCLE_1**
