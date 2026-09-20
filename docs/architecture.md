# Manuscript AI Architecture

The system uses a highly decoupled architecture isolating computer vision, OCR, and natural language processing (NLP) to reconstruct historical texts (Tamil and Arabic).

## Module Responsibilities
*   **app/**: Contains the main application entry points and REST API routers. Includes Streamlit widgets for validation.
*   **checkpoints/**: Dedicated storage for trained model weights. Segregated by model type (CRNN vs MLM).
*   **data/**: The isolated dataset hierarchy, physically separating OCR images from linguistic text corpora.
*   **models/**: Core ML engines.
    *   **cv/**: Image preprocessing, normalization, and line/word segmentation.
    *   **digitalization/**: Formats the raw OCR output into language-agnostic data structures with confidence scoring.
    *   **language_detection/**: Determines if the input is Tamil or Arabic and identifies the reading direction (LTR/RTL).
    *   **language_understanding/**: The NLP engine. Responsible for contextual embeddings, Masked Language Modeling (MLM), and grammar validation.
    *   **recognition/**: The primary CRNN OCR engine (Encoder, BiLSTM, CTC Decoder).
    *   **reconstruction/**: Orchestrates error detection and context-aware word replacement.
    *   **timeline/**: Historical classification models.
    *   **translation/**: Translates the final reconstructed text into English.
*   **pipeline/**: Orchestrators that wire the models together. The `manuscript_pipeline` calls the `language_router`, which passes execution to the respective `tamil_pipeline` or `arabic_pipeline`.
*   **scripts/**: Standalone ML operations (dataset prep, training loops, manifest generation).

## Data and Model Flow
1. Image enters via `app/api/inference.py`.
2. Routed to `pipeline/manuscript_pipeline.py`.
3. Preprocessed by `models/cv`.
4. Language determined by `models/language_detection`.
5. Routed to `TamilPipeline` or `ArabicPipeline`.
6. OCR performed by `models/recognition/crnn.py`.
7. Output formatted by `models/digitalization`.
8. Errors identified by `models/reconstruction/gap_detector.py`.
9. Sentences passed to `models/language_understanding/tamil_language_model.py` for MLM candidate generation.
10. Final word selected by `reconstructor.py`.
11. Results stored in `outputs/`.
