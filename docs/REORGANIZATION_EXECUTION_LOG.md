# Phase 1 Reorganization Execution Log

2026-09-19T16:24:00+05:30

## Moved
AUDIT_REPORT.md → docs\audits\
PROJECT_STATUS.md → docs\audits\
REFINEMENT_BASELINE.md → docs\audits\
CTC_IMPLEMENTATION_EVIDENCE.txt → docs\architecture\
PPT_EVIDENCE.txt → docs\architecture\
BEST_RUN_ANALYSIS.txt → docs\training\
BEST_BASELINE_TRAINING_HISTORY.csv → outputs\training\full_tamil_baseline\
BEST_BASELINE_TRAINING_HISTORY.json → outputs\training\full_tamil_baseline\
BEST_40_EPOCH_RESULTS.txt → outputs\training\global_shuffle_40epoch\
current_files.txt → outputs\archive\
ckpt_keys.txt → outputs\archive\
diagnostic_detailed.json → outputs\archive\diagnostics\
scratch_diag_out.json → outputs\archive\diagnostics\

## Skipped
sample.jpg
Reason: SKIPPED — active dependency found (in scripts\evaluate_autocrop.py, scripts\synthetic_segmentation_test.py, etc.)

sample_test.png
Reason: SKIPPED — active dependency found (in scripts\test_frontend.py, scripts\geometry_analysis.py, etc.)

sample_10lines.png
Reason: SKIPPED — active dependency found (in scripts\debug_multiline.py, scripts\eval_inference_ocr_temporal_fix.py, etc.)

synthetic_test_line.png
Reason: SKIPPED — active dependency found (in scripts\synthetic_segmentation_test.py)

## Protected
The following critical paths and files were completely untouched and NOT modified:
- checkpoints\
- data\tamil_ocr_dataset\
- data\TamilNet_old\
- data\text_corpus\
- models\
- app\
- scripts\
- venv\
- requirements.txt

## Verification
- Number of files moved: 13
- Number skipped: 4
- Number deleted: 0
- Whether any checkpoint hash changed: No. The hash of `checkpoints\recognition\tamil\ocr_balanced_temporal\best.pth` was verified as `07554040440E80E5EC2B1BD6EA1FA5BB6618E105AE3AD45F7590F21113E2F136` before and after.
- Whether requirements.txt changed: No
- Whether app source changed: No
- Whether model source changed: No
