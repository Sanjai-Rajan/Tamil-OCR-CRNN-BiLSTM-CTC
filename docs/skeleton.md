# Manuscript AI - Complete System Skeleton & Use Cases

This document outlines the complete folder and file structure for the `manuscript_ai` project, including the fully decoupled architecture isolating computer vision, OCR, and natural language processing, along with the specific use cases of each module.

```text
manuscript_ai/
│
├── data/
│   │
│   ├── raw/
│   │   ├── tamil/
│   │   └── arabic/
│   │
│   ├── processed/
│   │   ├── tamil/
│   │   └── arabic/
│   │
│   ├── segmented/
│   │   ├── tamil/
│   │   │   ├── lines/
│   │   │   └── words/
│   │   │
│   │   └── arabic/
│   │       ├── lines/
│   │       └── words/
│   │
│   ├── annotations/
│   │   ├── tamil/
│   │   └── arabic/
│   │
│   ├── tamil_corpus/
│   │   ├── raw/
│   │   ├── cleaned/
│   │   ├── tokenized/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   │
│   └── arabic_corpus/
│       ├── raw/
│       ├── cleaned/
│       ├── tokenized/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── models/
│   │
│   ├── cv/
│   │   ├── __init__.py
│   │   ├── preprocessor.py
│   │   ├── image_quality.py
│   │   └── segmenter.py
│   │
│   ├── language_detection/
│   │   ├── __init__.py
│   │   ├── script_detector.py
│   │   ├── language_detector.py
│   │   └── direction_detector.py
│   │
│   ├── recognition/
│   │   ├── __init__.py
│   │   ├── crnn.py
│   │   ├── encoder.py
│   │   ├── lstm.py
│   │   ├── decoder.py
│   │   └── loss.py
│   │
│   ├── digitalization/
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   ├── word_builder.py
│   │   ├── sentence_builder.py
│   │   └── confidence.py
│   │
│   ├── reconstruction/
│   │   ├── __init__.py
│   │   ├── gap_detector.py
│   │   ├── error_detector.py
│   │   ├── candidate_generator.py
│   │   ├── candidate_ranker.py
│   │   └── reconstructor.py
│   │
│   ├── language_understanding/
│   │   ├── __init__.py
│   │   ├── tamil_preprocessor.py
│   │   ├── tamil_tokenizer.py
│   │   ├── tamil_language_model.py
│   │   ├── context_manager.py
│   │   ├── paragraph_context.py
│   │   └── semantic_validator.py
│   │
│   ├── translation/
│   │   ├── __init__.py
│   │   ├── translator.py
│   │   └── translation_validator.py
│   │
│   └── timeline/
│       ├── __init__.py
│       ├── feature_extractor.py
│       ├── classifier.py
│       └── predictor.py
│
├── pipeline/
│   ├── __init__.py
│   ├── manuscript_pipeline.py
│   ├── language_router.py
│   ├── tamil_pipeline.py
│   ├── arabic_pipeline.py
│   └── result_manager.py
│
├── scripts/
│   ├── dataset_loader.py
│   ├── prepare_ocr_dataset.py
│   ├── prepare_language_dataset.py
│   ├── train_crnn.py
│   ├── train_language_model.py
│   ├── train_reconstruction.py
│   └── evaluate.py
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── inference.py
│   │
│   └── components/
│       ├── __init__.py
│       ├── image_viewer.py
│       ├── ocr_viewer.py
│       ├── reconstruction_viewer.py
│       └── validation_widget.py
│
├── outputs/
│   ├── raw_ocr/
│   ├── reconstructed/
│   ├── translations/
│   └── final/
│
├── checkpoints/
│   ├── recognition/
│   ├── tamil_language_model/
│   └── reconstruction/
│
├── tests/
│   ├── test_inference.py
│   ├── test_decoder.py
│   ├── test_encoder.py
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   ├── test_language_detection.py
│   ├── test_direction_detection.py
│   ├── test_digitalization.py
│   ├── test_tamil_preprocessing.py
│   ├── test_tamil_tokenizer.py
│   ├── test_language_model.py
│   ├── test_reconstruction.py
│   ├── test_translation.py
│   └── test_pipeline.py
│
├── config/
│   ├── model_config.yaml
│   ├── pipeline_config.yaml
│   └── language_config.yaml
│
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── metrics.py
│   ├── file_utils.py
│   └── device.py
│
├── database/
│   ├── __init__.py
│   ├── database.py
│   ├── schema.py
│   └── storage.py
│
├── notebooks/
│
├── docs/
│   ├── architecture.md
│   ├── pipeline.md
│   ├── dataset.md
│   ├── skeleton.md
│   └── error_log.md
│
└── requirements.txt
```

---

## Specific Use Cases by Component

### `data/tamil_corpus/` & `data/arabic_corpus/`
- **Use Case:** To train the contextual language model on actual grammar and literary vocabulary, completely independent of the visual manuscript images. Prevents the language model from simply memorizing the OCR training set.

### `models/cv/` (Computer Vision)
- **Use Case:** A faded, low-contrast manuscript image is enhanced for the OCR model (`preprocessor.py`). Word bounding boxes are extracted on a non-linear palm leaf (`segmenter.py`).

### `models/language_detection/` (Script & Language Routing)
- **Use Case:** A user uploads a mixed document. `script_detector.py` ensures Arabic segments go to the Arabic pipeline, while `direction_detector.py` determines the reading order (LTR/RTL) so CNN feature maps can be correctly processed.

### `models/recognition/` (OCR Engine)
- **Use Case:** To read the pixels of the manuscript and output the raw characters. *This module knows nothing about grammar; it only identifies visual shapes.* Maps visual features directly to character sequences. Outputs raw text containing potential errors or gaps.

### `models/digitalization/`
- **Use Case:** Creating a structured JSON representation of a manuscript page, grouping recognized characters into words and sentences with confidence scores (`tokenizer.py`, `word_builder.py`), preparing it for the language understanding module.

### `models/language_understanding/tamil_preprocessor.py`
- **Use Case:** To clean the raw Tamil corpus text before training (e.g., standardizing different Unicode representations of "க்ஷ").

### `models/language_understanding/tamil_tokenizer.py`
- **Use Case:** Correctly splits Tamil text into subwords/words so the model can process it without destroying complex combining characters. Provides a `[MASK]` token for corrupted areas.

### `models/language_understanding/tamil_language_model.py` (Masked Language Model)
- **Use Case:** When the OCR outputs "அவன் அரண்ம_னை சென்றான்", this module uses a transformer to suggest candidates like "அரண்மனைக்கு" (0.91) and "அரண்மனையை" (0.05).

### `models/language_understanding/paragraph_context.py`
- **Use Case:** If a missing word could be "king" or "queen" based on grammar alone, this module looks at the *previous sentence* (or paragraph context) to see who the subject is, resolving the ambiguity.

### `models/reconstruction/reconstructor.py`
- **Use Case:** The "manager" of the NLP phase. It takes the structured digitalized output, checks if there are errors via `error_detector.py` and `gap_detector.py`, asks the language model for candidates, applies contextual logic, and formats the final JSON output (Original, Reconstructed, Confidence, Alternatives).

### `models/reconstruction/candidate_ranker.py`
- **Use Case:** An OCR outputs "b_at" (gap). The candidate generator suggests "boat" or "bat". The ranker chooses "boat" because the paragraph context (`paragraph_context.py`) is about "fishing".

### `models/translation/translator.py`
- **Use Case:** Making ancient texts accessible to international researchers by performing specialized machine translation (e.g., Classical Tamil to English) on the fully reconstructed digital text.

### `models/timeline/`
- **Use Case:** Identifying that a specific curve in a Tamil letter places the manuscript in the 12th century Chola period, analyzing script evolution and vocabulary usage.

### `pipeline/` (Orchestrators)
- **Use Case:** The `manuscript_pipeline.py` receives the input image, asks `language_router.py` what language it is, and then directs traffic to the specialized `tamil_pipeline.py` or `arabic_pipeline.py` which executes the sequence: CV -> Recognition -> Reconstruction -> Digitalization.

### `app/main.py` & `components/`
- **Use Case:** Displays the manuscript to historians. If the system is uncertain about a reconstruction (e.g., confidence < 0.8), it renders a `validation_widget.py` with `[Accept]`, `[Reject]`, and `[Edit]` buttons so human experts have the final say.
