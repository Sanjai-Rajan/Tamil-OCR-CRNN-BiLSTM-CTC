# Script Dependency Audit

## Inventory

### analyze_char_dataset.py
- **Path**: scripts/analyze_char_dataset.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### analyze_ctc_blank_behavior.py
- **Path**: scripts/analyze_ctc_blank_behavior.py
- **Internal Imports**: ['models.recognition.decoder', 'scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### audit_datasets.py
- **Path**: scripts/audit_datasets.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### build_atomic_character_dataset.py
- **Path**: scripts/build_atomic_character_dataset.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### build_tamil_vocabulary.py
- **Path**: scripts/build_tamil_vocabulary.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### compare_ocr_results.py
- **Path**: scripts/compare_ocr_results.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### component_isolation_diagnostic.py
- **Path**: scripts/component_isolation_diagnostic.py
- **Internal Imports**: ['models.recognition.decoder', 'scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### convert_tamilnet.py
- **Path**: scripts/convert_tamilnet.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### cropping_analysis.py
- **Path**: scripts/cropping_analysis.py
- **Internal Imports**: ['models.recognition.decoder', 'scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: ['sample_test.png']


### crop_validation.py
- **Path**: scripts/crop_validation.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### dataset_loader.py
- **Path**: scripts/dataset_loader.py
- **Internal Imports**: []
- **Imported by other scripts**: ['analyze_ctc_blank_behavior.py', 'component_isolation_diagnostic.py', 'cropping_analysis.py', 'diagnose_ctc.py', 'dynamic_width_diagnostic.py', 'evaluate_autocrop.py', 'evaluate_crnn.py', 'evaluate_custom.py', 'evaluate_exp_A.py', 'evaluate_ocr.py', 'evaluate_scale_response.py', 'eval_crnn.py', 'eval_test_balanced.py', 'forensic_audit_sampling.py', 'offline_diagnostic.py', 'preview_exp_a2.py', 'preview_multiscale.py', 'root_cause_diagnostic.py', 'root_cause_diagnostic_extended.py', 'scale_diagnostic.py', 'single_batch_diagnostic.py', 'test_architecture.py', 'tiny_overfit_diagnostic.py', 'train_crnn.py', 'train_overfit_tiny.py', 'train_refinement.py', 'validate_bucketing.py']
- **Imported by active code (app/models)**: [('.\\tiny_overfit.py', 'scripts.dataset_loader'), ('.\\baseline_before_character_stage\\scripts\\train_crnn.py', 'scripts.dataset_loader'), ('.\\scripts\\preview_exp_a2.py', 'scripts.dataset_loader'), ('.\\scripts\\eval_test_balanced.py', 'scripts.dataset_loader'), ('.\\scripts\\evaluate_scale_response.py', 'scripts.dataset_loader'), ('.\\temporal_analysis.py', 'scripts.dataset_loader'), ('.\\scripts\\train_crnn.py', 'scripts.dataset_loader'), ('.\\scripts\\diagnose_ctc.py', 'scripts.dataset_loader'), ('.\\scratch_diagnose_t.py', 'scripts.dataset_loader'), ('.\\baseline_before_character_stage\\scripts\\evaluate_custom.py', 'scripts.dataset_loader'), ('.\\scripts\\evaluate_custom.py', 'scripts.dataset_loader'), ('.\\scripts\\cropping_analysis.py', 'scripts.dataset_loader'), ('.\\forensic_analysis.py', 'scripts.dataset_loader'), ('.\\baseline_before_character_stage\\scripts\\root_cause_diagnostic.py', 'scripts.dataset_loader'), ('.\\scratch\\root_cause_diagnostic.py', 'scripts.dataset_loader'), ('.\\scripts\\analyze_ctc_blank_behavior.py', 'scripts.dataset_loader'), ('.\\scripts\\root_cause_diagnostic.py', 'scripts.dataset_loader'), ('.\\scripts\\train_refinement.py', 'scripts.dataset_loader'), ('.\\scripts\\tiny_overfit_diagnostic.py', 'scripts.dataset_loader'), ('.\\baseline_before_character_stage\\scripts\\diagnose_ctc.py', 'scripts.dataset_loader'), ('.\\scripts\\test_architecture.py', 'scripts.dataset_loader'), ('.\\scripts\\single_batch_diagnostic.py', 'scripts.dataset_loader'), ('.\\baseline_before_character_stage\\scripts\\evaluate_crnn.py', 'scripts.dataset_loader'), ('.\\baseline_before_character_stage\\scripts\\dynamic_width_diagnostic.py', 'scripts.dataset_loader'), ('.\\scripts\\forensic_audit_sampling.py', 'scripts.dataset_loader'), ('.\\tests\\test_training.py', 'scripts.dataset_loader'), ('.\\baseline_before_character_stage\\scripts\\root_cause_diagnostic_extended.py', 'scripts.dataset_loader'), ('.\\scripts\\evaluate_autocrop.py', 'scripts.dataset_loader'), ('.\\baseline_before_character_stage\\scripts\\component_isolation_diagnostic.py', 'scripts.dataset_loader'), ('.\\scripts\\root_cause_diagnostic_extended.py', 'scripts.dataset_loader'), ('.\\preflight.py', 'scripts.dataset_loader'), ('.\\scripts\\evaluate_crnn.py', 'scripts.dataset_loader'), ('.\\scripts\\offline_diagnostic.py', 'scripts.dataset_loader'), ('.\\scripts\\train_overfit_tiny.py', 'scripts.dataset_loader'), ('.\\scripts\\evaluate_exp_A.py', 'scripts.dataset_loader'), ('.\\scripts\\dynamic_width_diagnostic.py', 'scripts.dataset_loader'), ('.\\scripts\\preview_multiscale.py', 'scripts.dataset_loader'), ('.\\scripts\\component_isolation_diagnostic.py', 'scripts.dataset_loader'), ('.\\tests\\test_dataset_loader.py', 'scripts.dataset_loader'), ('.\\scratch_diagnose_json.py', 'scripts.dataset_loader'), ('.\\scratch_preflight.py', 'scripts.dataset_loader'), ('.\\scripts\\eval_crnn.py', 'scripts.dataset_loader'), ('.\\scripts\\validate_bucketing.py', 'scripts.dataset_loader'), ('.\\scratch_diagnostic_detailed.py', 'scripts.dataset_loader'), ('.\\baseline_before_character_stage\\scripts\\tiny_overfit_diagnostic.py', 'scripts.dataset_loader'), ('.\\scripts\\evaluate_ocr.py', 'scripts.dataset_loader'), ('.\\baseline_before_character_stage\\scripts\\single_batch_diagnostic.py', 'scripts.dataset_loader'), ('.\\scratch_diagnose.py', 'scripts.dataset_loader'), ('.\\scripts\\scale_diagnostic.py', 'scripts.dataset_loader')]
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### dataset_status.py
- **Path**: scripts/dataset_status.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### debug_multiline.py
- **Path**: scripts/debug_multiline.py
- **Internal Imports**: ['models.recognition.decoder', 'app.inference', 'models.cv.segmenter']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: ['sample_10lines.png']


### diagnose_ctc.py
- **Path**: scripts/diagnose_ctc.py
- **Internal Imports**: ['scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### ds3_audit.py
- **Path**: scripts/ds3_audit.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### ds6_audit.py
- **Path**: scripts/ds6_audit.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### ds6_copy.py
- **Path**: scripts/ds6_copy.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### ds6_integrate.py
- **Path**: scripts/ds6_integrate.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### ds6_validate_splits.py
- **Path**: scripts/ds6_validate_splits.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### ds7_audit.py
- **Path**: scripts/ds7_audit.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### ds_combined_audit.py
- **Path**: scripts/ds_combined_audit.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### ds_pre5090_audit.py
- **Path**: scripts/ds_pre5090_audit.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### ds_uar_audit.py
- **Path**: scripts/ds_uar_audit.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### ds_uar_generate_reports.py
- **Path**: scripts/ds_uar_generate_reports.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### ds_uar_second_stage_audit.py
- **Path**: scripts/ds_uar_second_stage_audit.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### dynamic_width_diagnostic.py
- **Path**: scripts/dynamic_width_diagnostic.py
- **Internal Imports**: ['scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### evaluate.py
- **Path**: scripts/evaluate.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### evaluate_autocrop.py
- **Path**: scripts/evaluate_autocrop.py
- **Internal Imports**: ['models.recognition.decoder', 'scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: ['sample.jpg', 'sample_test.png']


### evaluate_crnn.py
- **Path**: scripts/evaluate_crnn.py
- **Internal Imports**: ['models.recognition.decoder', 'scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### evaluate_custom.py
- **Path**: scripts/evaluate_custom.py
- **Internal Imports**: ['models.recognition.crnn', 'models.digitalization.tokenizer', 'scripts.dataset_loader', 'models.recognition.decoder']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### evaluate_exp_A.py
- **Path**: scripts/evaluate_exp_A.py
- **Internal Imports**: ['models.recognition.decoder', 'scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: ['sample_test.png']


### evaluate_ocr.py
- **Path**: scripts/evaluate_ocr.py
- **Internal Imports**: ['models.recognition.crnn', 'models.digitalization.tokenizer', 'scripts.dataset_loader', 'models.recognition.decoder']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: True
- **References samples**: []


### evaluate_scale_response.py
- **Path**: scripts/evaluate_scale_response.py
- **Internal Imports**: ['models.recognition.decoder', 'scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: ['sample_test.png']


### eval_crnn.py
- **Path**: scripts/eval_crnn.py
- **Internal Imports**: ['models.recognition.decoder', 'scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### eval_epoch11.py
- **Path**: scripts/eval_epoch11.py
- **Internal Imports**: ['models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### eval_inference_balanced.py
- **Path**: scripts/eval_inference_balanced.py
- **Internal Imports**: ['models.recognition.decoder', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: ['sample.jpg', 'sample_test.png', 'sample_10lines.png']


### eval_inference_ocr_temporal_fix.py
- **Path**: scripts/eval_inference_ocr_temporal_fix.py
- **Internal Imports**: ['models.recognition.decoder', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: ['sample.jpg', 'sample_test.png', 'sample_10lines.png']


### eval_test_balanced.py
- **Path**: scripts/eval_test_balanced.py
- **Internal Imports**: ['models.recognition.decoder', 'scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### extract_error_annotated.py
- **Path**: scripts/extract_error_annotated.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: True
- **References samples**: []


### extract_project_madurai.py
- **Path**: scripts/extract_project_madurai.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### extract_sentamizh.py
- **Path**: scripts/extract_sentamizh.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### extract_tamil_lm_corpus.py
- **Path**: scripts/extract_tamil_lm_corpus.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### forensic_analysis.py
- **Path**: scripts/forensic_analysis.py
- **Internal Imports**: ['models.recognition.decoder', 'app.inference', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### forensic_audit_sampling.py
- **Path**: scripts/forensic_audit_sampling.py
- **Internal Imports**: ['models.recognition.decoder', 'scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### forensic_segmentation.py
- **Path**: scripts/forensic_segmentation.py
- **Internal Imports**: ['models.cv.segmenter']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: ['sample.jpg', 'sample_test.png', 'sample_10lines.png']


### generate_manifests.py
- **Path**: scripts/generate_manifests.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### geometry_analysis.py
- **Path**: scripts/geometry_analysis.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: ['sample.jpg', 'sample_test.png']


### global_char_inventory.py
- **Path**: scripts/global_char_inventory.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### import_all_packets.py
- **Path**: scripts/import_all_packets.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### import_tamilnet.py
- **Path**: scripts/import_tamilnet.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### import_tamilnet_converter.py
- **Path**: scripts/import_tamilnet_converter.py
- **Internal Imports**: ['models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### import_tamil_packet.py
- **Path**: scripts/import_tamil_packet.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: [('.\\tests\\test_dataset.py', 'scripts.import_tamil_packet')]
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### integrate_ds2_ds3.py
- **Path**: scripts/integrate_ds2_ds3.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### map_tamilchar_atomic.py
- **Path**: scripts/map_tamilchar_atomic.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### measure_width_distribution.py
- **Path**: scripts/measure_width_distribution.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### mlm_checkpoint_test.py
- **Path**: scripts/mlm_checkpoint_test.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### mlm_smoke_test.py
- **Path**: scripts/mlm_smoke_test.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### monitor_mlm_live.py
- **Path**: scripts/monitor_mlm_live.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: True
- **References samples**: []


### multi_scale_segmentation_test.py
- **Path**: scripts/multi_scale_segmentation_test.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: ['sample_10lines.png']


### offline_diagnostic.py
- **Path**: scripts/offline_diagnostic.py
- **Internal Imports**: ['models.recognition.decoder', 'scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: ['sample_test.png']


### prepare_character_dataset.py
- **Path**: scripts/prepare_character_dataset.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### prepare_language_dataset.py
- **Path**: scripts/prepare_language_dataset.py
- **Internal Imports**: ['models.language_understanding.tamil_preprocessor']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### prepare_ocr_dataset.py
- **Path**: scripts/prepare_ocr_dataset.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### prepare_tamil_lm_corpus.py
- **Path**: scripts/prepare_tamil_lm_corpus.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### prepare_tamil_ocr_dataset.py
- **Path**: scripts/prepare_tamil_ocr_dataset.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### preprocess_project_madurai.py
- **Path**: scripts/preprocess_project_madurai.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### preview_exp_a2.py
- **Path**: scripts/preview_exp_a2.py
- **Internal Imports**: ['models.recognition.decoder', 'scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### preview_multiscale.py
- **Path**: scripts/preview_multiscale.py
- **Internal Imports**: ['scripts.dataset_loader', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### regenerate_ds3.py
- **Path**: scripts/regenerate_ds3.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### root_cause_diagnostic.py
- **Path**: scripts/root_cause_diagnostic.py
- **Internal Imports**: ['scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### root_cause_diagnostic_extended.py
- **Path**: scripts/root_cause_diagnostic_extended.py
- **Internal Imports**: ['scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### run_a100_experiment.py
- **Path**: scripts/run_a100_experiment.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### sanity_check.py
- **Path**: scripts/sanity_check.py
- **Internal Imports**: ['models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### scale_diagnostic.py
- **Path**: scripts/scale_diagnostic.py
- **Internal Imports**: ['models.recognition.decoder', 'scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### single_batch_diagnostic.py
- **Path**: scripts/single_batch_diagnostic.py
- **Internal Imports**: ['models.recognition.decoder', 'scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### split_dataset.py
- **Path**: scripts/split_dataset.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### synthetic_segmentation_test.py
- **Path**: scripts/synthetic_segmentation_test.py
- **Internal Imports**: ['app.inference']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: ['sample.jpg', 'sample_test.png', 'synthetic_test_line.png']


### test_architecture.py
- **Path**: scripts/test_architecture.py
- **Internal Imports**: ['scripts.dataset_loader', 'models.reconstruction.restoration_model', 'models.recognition.uncertainty', 'models.recognition.crnn', 'models.reconstruction.ranker', 'models.digitalization.tokenizer', 'models.language_understanding.tamil_language_model']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### test_frontend.py
- **Path**: scripts/test_frontend.py
- **Internal Imports**: ['app.inference']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: ['sample_test.png']


### tiny_overfit_diagnostic.py
- **Path**: scripts/tiny_overfit_diagnostic.py
- **Internal Imports**: ['models.recognition.crnn', 'models.digitalization.tokenizer', 'scripts.dataset_loader', 'models.recognition.decoder']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### train_character_classifier.py
- **Path**: scripts/train_character_classifier.py
- **Internal Imports**: ['models.recognition.character_classifier']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### train_crnn.py
- **Path**: scripts/train_crnn.py
- **Internal Imports**: ['models.recognition.crnn', 'models.digitalization.tokenizer', 'scripts.dataset_loader', 'models.recognition.decoder']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### train_overfit_tiny.py
- **Path**: scripts/train_overfit_tiny.py
- **Internal Imports**: ['models.recognition.decoder', 'scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### train_refinement.py
- **Path**: scripts/train_refinement.py
- **Internal Imports**: ['models.recognition.crnn', 'models.digitalization.tokenizer', 'scripts.dataset_loader', 'models.recognition.decoder']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### validate_bucketing.py
- **Path**: scripts/validate_bucketing.py
- **Internal Imports**: ['scripts.dataset_loader', 'models.recognition.crnn', 'models.digitalization.tokenizer']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: True
- **Hardcoded path assumptions**: False
- **References samples**: []


### validate_ctc_length.py
- **Path**: scripts/validate_ctc_length.py
- **Internal Imports**: ['models.recognition.crnn']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### validate_dataset.py
- **Path**: scripts/validate_dataset.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### validate_tamil_dataset.py
- **Path**: scripts/validate_tamil_dataset.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### verify_ckpt.py
- **Path**: scripts/verify_ckpt.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### verify_experiment_b.py
- **Path**: scripts/verify_experiment_b.py
- **Internal Imports**: ['models.recognition.crnn', 'models.recognition.crnn_v2']
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


### __init__.py
- **Path**: scripts/__init__.py
- **Internal Imports**: []
- **Imported by other scripts**: []
- **Imported by active code (app/models)**: []
- **Imports scripts.dataset_loader**: False
- **Hardcoded path assumptions**: False
- **References samples**: []


## Prohibited AI Artifacts Search

- File: generate_manifests.py | Term: language model | Context: JSONL manifests for language modeling."""         pass      
- File: mlm_checkpoint_test.py | Term: bert | Context: / "models" / "indic_bert" TEST_OUT_DIR = PROJECT_ROOT / "out
- File: mlm_checkpoint_test.py | Term: transformers | Context: import Dataset from transformers import (     AutoTokenizer,
- File: mlm_checkpoint_test.py | Term: AutoModel | Context:  AutoTokenizer,     AutoModelForMaskedLM,     DataCollatorFo
- File: mlm_checkpoint_test.py | Term: AutoTokenizer | Context: ormers import (     AutoTokenizer,     AutoModelForMaskedLM,
- File: mlm_checkpoint_test.py | Term: from_pretrained | Context: zer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_fi
- File: mlm_checkpoint_test.py | Term: pretrained | Context:  AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_o
- File: mlm_smoke_test.py | Term: BERT | Context: rint(f"Loading IndicBERT from {model_dir}")     try:        
- File: mlm_smoke_test.py | Term: bert | Context: / "models" / "indic_bert"          print(f"Loading IndicBERT
- File: mlm_smoke_test.py | Term: IndicBERT | Context:     print(f"Loading IndicBERT from {model_dir}")     try:   
- File: mlm_smoke_test.py | Term: transformers | Context: h import torch from transformers import AutoTokenizer, AutoM
- File: mlm_smoke_test.py | Term: AutoModel | Context: port AutoTokenizer, AutoModelForMaskedLM  PROJECT_ROOT = Pat
- File: mlm_smoke_test.py | Term: AutoTokenizer | Context: transformers import AutoTokenizer, AutoModelForMaskedLM  PRO
- File: mlm_smoke_test.py | Term: from_pretrained | Context: zer = AutoTokenizer.from_pretrained(str(model_dir), local_fi
- File: mlm_smoke_test.py | Term: pretrained | Context:  AutoTokenizer.from_pretrained(str(model_dir), local_files_o
- File: monitor_mlm_live.py | Term: BERT | Context: ("       TAMIL INDICBERT MLM — LIVE TRAINING")             p
- File: monitor_mlm_live.py | Term: bert | Context: ts" / "mlm" / "indicbert_run_01" PROGRESS_FILE = RUN_DIR / "
- File: prepare_tamil_lm_corpus.py | Term: BERT | Context: ubword tokenizer (ALBERT tokenizer) specific to the IndicBER
- File: prepare_tamil_lm_corpus.py | Term: bert | Context: ed("ai4bharat/indic-bert")` - **Type:** Subword tokenizer (A
- File: prepare_tamil_lm_corpus.py | Term: IndicBERT | Context: er) specific to the IndicBERT model. - **Vocabulary Compatib
- File: prepare_tamil_lm_corpus.py | Term: transformers | Context: ace `datasets` and `transformers`. - **Loading Mechanism:** 
- File: prepare_tamil_lm_corpus.py | Term: AutoTokenizer | Context: s - **Tokenizer:** `AutoTokenizer.from_pretrained("ai4bharat
- File: prepare_tamil_lm_corpus.py | Term: from_pretrained | Context: r:** `AutoTokenizer.from_pretrained("ai4bharat/indic-bert")`
- File: prepare_tamil_lm_corpus.py | Term: pretrained | Context: `AutoTokenizer.from_pretrained("ai4bharat/indic-bert")` - **
- File: test_architecture.py | Term: mT5 | Context: ration_model import mT5RestorationModel from models.language
- File: test_architecture.py | Term: BERT | Context: nt("4. Loading IndicBERT...")     lm_dir = PROJECT_ROOT / co
- File: test_architecture.py | Term: IndicBERT | Context:   print("4. Loading IndicBERT...")     lm_dir = PROJECT_ROOT
