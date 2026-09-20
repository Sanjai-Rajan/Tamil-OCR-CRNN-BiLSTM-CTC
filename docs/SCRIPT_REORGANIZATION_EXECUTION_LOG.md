# Phase 2 Script Reorganization Execution Log
2026-09-19T16:28:00+05:30

## Dependency Audit
Total Python scripts: 91
Active: 88
Historical: 3
Unknown: 0
Potential prohibited-AI references: Found in mlm_smoke_test, prepare_tamil_lm_corpus, test_architecture, etc. (Logged in SCRIPT_DEPENDENCY_AUDIT.md)

## Files Moved
scripts\analyze_char_dataset.py -> scripts\data\analyze_char_dataset.py
scripts\audit_datasets.py -> scripts\diagnostics\audit_datasets.py
scripts\build_atomic_character_dataset.py -> scripts\data\build_atomic_character_dataset.py
scripts\build_tamil_vocabulary.py -> scripts\data\build_tamil_vocabulary.py
scripts\compare_ocr_results.py -> scripts\evaluation\compare_ocr_results.py
scripts\convert_tamilnet.py -> scripts\data\convert_tamilnet.py
scripts\crop_validation.py -> scripts\historical\crop_validation.py
scripts\dataset_status.py -> scripts\data\dataset_status.py
scripts\ds3_audit.py -> scripts\diagnostics\ds3_audit.py
scripts\ds6_audit.py -> scripts\diagnostics\ds6_audit.py
scripts\ds6_copy.py -> scripts\historical\ds6_copy.py
scripts\ds6_integrate.py -> scripts\historical\ds6_integrate.py
scripts\ds6_validate_splits.py -> scripts\diagnostics\ds6_validate_splits.py
scripts\ds7_audit.py -> scripts\diagnostics\ds7_audit.py
scripts\ds_combined_audit.py -> scripts\diagnostics\ds_combined_audit.py
scripts\ds_pre5090_audit.py -> scripts\diagnostics\ds_pre5090_audit.py
scripts\ds_uar_audit.py -> scripts\diagnostics\ds_uar_audit.py
scripts\ds_uar_generate_reports.py -> scripts\historical\ds_uar_generate_reports.py
scripts\ds_uar_second_stage_audit.py -> scripts\diagnostics\ds_uar_second_stage_audit.py
scripts\evaluate.py -> scripts\evaluation\evaluate.py
scripts\extract_project_madurai.py -> scripts\historical\extract_project_madurai.py
scripts\extract_sentamizh.py -> scripts\historical\extract_sentamizh.py
scripts\extract_tamil_lm_corpus.py -> scripts\data\extract_tamil_lm_corpus.py
scripts\generate_manifests.py -> scripts\historical\generate_manifests.py
scripts\global_char_inventory.py -> scripts\historical\global_char_inventory.py
scripts\import_all_packets.py -> scripts\data\import_all_packets.py
scripts\import_tamilnet.py -> scripts\data\import_tamilnet.py
scripts\import_tamil_packet.py -> scripts\data\import_tamil_packet.py
scripts\integrate_ds2_ds3.py -> scripts\historical\integrate_ds2_ds3.py
scripts\map_tamilchar_atomic.py -> scripts\historical\map_tamilchar_atomic.py
scripts\measure_width_distribution.py -> scripts\historical\measure_width_distribution.py
scripts\mlm_checkpoint_test.py -> scripts\historical\mlm_checkpoint_test.py
scripts\mlm_smoke_test.py -> scripts\historical\mlm_smoke_test.py
scripts\prepare_character_dataset.py -> scripts\data\prepare_character_dataset.py
scripts\prepare_ocr_dataset.py -> scripts\data\prepare_ocr_dataset.py
scripts\prepare_tamil_lm_corpus.py -> scripts\data\prepare_tamil_lm_corpus.py
scripts\prepare_tamil_ocr_dataset.py -> scripts\data\prepare_tamil_ocr_dataset.py
scripts\preprocess_project_madurai.py -> scripts\historical\preprocess_project_madurai.py
scripts\regenerate_ds3.py -> scripts\historical\regenerate_ds3.py
scripts\run_a100_experiment.py -> scripts\training\run_a100_experiment.py
scripts\split_dataset.py -> scripts\data\split_dataset.py
scripts\validate_dataset.py -> scripts\diagnostics\validate_dataset.py
scripts\validate_tamil_dataset.py -> scripts\diagnostics\validate_tamil_dataset.py
scripts\verify_ckpt.py -> scripts\diagnostics\verify_ckpt.py

## Files Skipped
analyze_ctc_blank_behavior.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

component_isolation_diagnostic.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

cropping_analysis.py
Reason: Active sample dependency: ['sample_test.png']

debug_multiline.py
Reason: Active sample dependency: ['sample_10lines.png']

diagnose_ctc.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

dynamic_width_diagnostic.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

evaluate_autocrop.py
Reason: Active sample dependency: ['sample.jpg', 'sample_test.png']

evaluate_crnn.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

evaluate_custom.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

evaluate_exp_A.py
Reason: Active sample dependency: ['sample_test.png']

evaluate_ocr.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

evaluate_scale_response.py
Reason: Active sample dependency: ['sample_test.png']

eval_crnn.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

eval_epoch11.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

eval_inference_balanced.py
Reason: Active sample dependency: ['sample.jpg', 'sample_test.png', 'sample_10lines.png']

eval_inference_ocr_temporal_fix.py
Reason: Active sample dependency: ['sample.jpg', 'sample_test.png', 'sample_10lines.png']

eval_test_balanced.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

extract_error_annotated.py
Reason: Hardcoded path assumptions (C:\...)

forensic_analysis.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

forensic_audit_sampling.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

forensic_segmentation.py
Reason: Active sample dependency: ['sample.jpg', 'sample_test.png', 'sample_10lines.png']

geometry_analysis.py
Reason: Active sample dependency: ['sample.jpg', 'sample_test.png']

import_tamilnet_converter.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

monitor_mlm_live.py
Reason: Hardcoded path assumptions (C:\...)

multi_scale_segmentation_test.py
Reason: Active sample dependency: ['sample_10lines.png']

offline_diagnostic.py
Reason: Active sample dependency: ['sample_test.png']

prepare_language_dataset.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

preview_exp_a2.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

preview_multiscale.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

root_cause_diagnostic.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

root_cause_diagnostic_extended.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

sanity_check.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

scale_diagnostic.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

single_batch_diagnostic.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

synthetic_segmentation_test.py
Reason: Active sample dependency: ['sample.jpg', 'sample_test.png', 'synthetic_test_line.png']

test_architecture.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

test_frontend.py
Reason: Active sample dependency: ['sample_test.png']

tiny_overfit_diagnostic.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

train_character_classifier.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

train_overfit_tiny.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

train_refinement.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

validate_bucketing.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

validate_ctc_length.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

verify_experiment_b.py
Reason: Internal imports (models/scripts/app) would break if moved without sys.path refactoring

## Protected
Explicitly list:
- scripts\dataset_loader.py
- scripts\train_crnn.py
- checkpoints\
- data\
- models\
- app\
- venv\
- requirements.txt

## Verification
- Python syntax/compile check: Passed
- import check: Passed (Unsafe scripts skipped)
- checkpoint existence: Verified
- checkpoint SHA-256: 07554040440E80E5EC2B1BD6EA1FA5BB6618E105AE3AD45F7590F21113E2F136
- requirements.txt unchanged: Yes
- app source unchanged: Yes
- model source unchanged: Yes
- dataset unchanged: Yes
- number of files deleted: 0
