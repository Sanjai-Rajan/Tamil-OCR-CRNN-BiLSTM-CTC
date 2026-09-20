# OCR Temporal Fix Resume Report

## Background
The `ocr_temporal_fix` experiment was paused after successfully completing Epoch 1. The checkpoint was saved at `checkpoints/recognition/tamil/ocr_temporal_fix/latest.pth` and the state was recorded in `outputs/training/tamil/ocr_temporal_fix/training_state.json`.

## Issue with Legacy Checkpoint
Upon attempting to resume, it was discovered that the original `train_crnn.py` implementation lacked correct state saving mechanisms for:
1. **Scheduler State**: The `OneCycleLR` scheduler state was never saved in the checkpoint.
2. **RNG State**: Python, NumPy, and PyTorch random number generator states were missing.

Additionally, the `OneCycleLR` schedule was calculated based on `epochs_to_add` rather than `total_epochs`, which would have caused a new cycle to be compressed into the remaining 9 epochs if resumed naïvely.

## Resolution

### 1. Script Fixes
The `train_crnn.py` script was updated to properly serialize and load the following states for all *future* checkpoints:
- `scheduler_state_dict`
- `rng_state` (Python `random`, `numpy`, and `torch` CPU/CUDA states)

We also introduced the `--total-epochs` parameter to ensure the `OneCycleLR` total steps are computed over the original 10-epoch lifespan instead of being compressed.

### 2. Legacy Checkpoint Reconstruction
For the existing Epoch 1 checkpoint:
- **Model and Optimizer**: Accurately restored from the `.pth` file.
- **OneCycleLR Scheduler**: Reconstructed by re-initializing the scheduler with `total_epochs=10` and manually stepping it `988` times (`126419 // 128 + 1` steps per epoch * 1 completed epoch). This placed the scheduler precisely at its expected end-of-Epoch-1 state.
- **RNG State**: Exact stochastic reproducibility from the end of Epoch 1 is mathematically impossible due to the missing data. A fresh RNG state was used for Epoch 2 onwards, which is an acceptable compromise to preserve the trained weights.

### 3. Execution
The training was resumed with:
- Target epochs: 10
- Batch size: 128
- Architecture: `ocr_temporal_fix` (CRNN with less_downsample)
- Resumed at: Epoch 2

No pretrained models (mT5, BERT, etc.) were introduced, and the model was strictly trained from scratch as required.
