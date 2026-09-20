# Project Status

## 1. Current best experiment:
`global_shuffle_lr_1e-4`

## 2. Dataset:
- 75,736 training samples
- 11,598 validation samples

## 3. Best checkpoint:
`checkpoints/recognition/tamil/global_shuffle_lr_1e-4/best.pth`

## 4. Best epoch:
66

## 5. Best validation metrics:
- CER = 0.57355
- Character Accuracy = 0.42645
- WER = 0.99543
- Word/Exact Match Accuracy = 0.00457
- Validation Loss = 2.16551
- Blank Prediction = 91.77%
- Average Prediction Length = 6.05
- Unique Predicted Characters = 57
- Unique Predicted Sequences = 9423

## 6. Training Status
Training continued to epoch 78, but epoch 66 remains the best validation CER.

## 7. Current interpretation:
The model is learning diverse Tamil character sequences but produces substantially shorter/incomplete predictions. Root cause has NOT yet been definitively established.

## 8. NEXT STEP:
Tomorrow perform a diagnostic of actual temporal length T versus ground-truth target length, inspect CNN downsampling, CTC feasibility, prediction-length distributions, and 50+ GT→prediction examples.

## 9. Important:
Do not make any changes based on assumptions.
Tomorrow's diagnostic must inspect the CURRENT implementation actually used by `global_shuffle_lr_1e-4`.
