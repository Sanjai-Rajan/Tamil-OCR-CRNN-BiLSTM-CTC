# OCR Balanced Temporal Current Status Report

## 1. Safety & Execution Status
- **Training Running**: No (Process successfully exited during the prior session block).
- **Process Exit Status**: Normal conclusion of scheduled block. The `OneCycleLR` hit its max step limit of `39,510` exactly at the end of Epoch 10, resulting in a safe halt (via `ValueError` during scheduler step for Epoch 11).
- **Training Resumable**: Yes, can be safely resumed. Checkpoints are intact.

## 2. Checkpoint State
Location: `checkpoints/recognition/tamil/ocr_balanced_temporal/`

### `latest.pth`
- **Existence**: Present
- **Timestamp**: `Fri Sep 18 08:24:21 2026`
- **Size**: 130,191,666 bytes
- **Epoch**: 10
- **`model_state_dict`**: Present
- **`optimizer_state_dict`**: Present
- **`scheduler_state_dict`**: Present
- **`rng_state`**: Present
- **Architecture Config**: Present

### `best.pth`
- **Existence**: Present
- **Timestamp**: `Fri Sep 18 08:24:21 2026`
- **Size**: 130,188,300 bytes
- **Epoch**: 10
- **`model_state_dict`**: Present
- **`optimizer_state_dict`**: Present
- **`scheduler_state_dict`**: Present
- **`rng_state`**: Present
- **Architecture Config**: Present

## 3. Training Progress & Metrics
- **Last Completed Epoch**: Epoch 10
- **Last Completed Batch**: Batch 3951 / 3951 (End of Epoch 10)
- **Partially Completed Epoch**: None (Epoch 11 did not process any batches)

### Measured Metrics (Epochs 6-10)
| Epoch | Train Loss | Validation Loss | CER | WER | Char Accuracy | Word Accuracy | Blank % | Avg Pred Length | Avg Target Length |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **6** | 1.2191 | 1.9536 | 0.5386 (53.86%) | 0.9956 (99.56%) | 0.4614 (46.14%) | 0.0044 (0.44%) | 93.11% | 5.08 tokens | ~4.79 tokens |
| **7** | 1.1678 | 1.9096 | 0.5170 (51.70%) | 0.9941 (99.41%) | 0.4830 (48.30%) | 0.0059 (0.59%) | 92.57% | 5.47 tokens | ~4.79 tokens |
| **8** | 1.1207 | 1.8791 | 0.5077 (50.77%) | 0.9939 (99.39%) | 0.4923 (49.23%) | 0.0061 (0.61%) | 92.35% | 5.65 tokens | ~4.79 tokens |
| **9** | 1.0787 | 1.8631 | 0.5055 (50.55%) | 0.9915 (99.15%) | 0.4945 (49.45%) | 0.0085 (0.85%) | 92.42% | 5.59 tokens | ~4.79 tokens |
| **10** | 1.0524 | 1.8618 | 0.5034 (50.34%) | 0.9908 (99.08%) | 0.4966 (49.66%) | 0.0092 (0.92%) | 92.31% | 5.66 tokens | ~4.79 tokens |

*(Note: Epoch 6, 7, 8, 9, and 10 all completed successfully. No metrics have been invented for Epoch 11).*
