# Dataset Infrastructure

The `manuscript_ai` project mandates three entirely independent ML pipelines, reflecting three separate dataset domains.

## 1. OCR Dataset
**Purpose:** Train the CRNN to transcribe image pixels into characters.
**Input:** Manuscript line/word crops.
**Target:** Ground truth transcription.
**Location:** `data/ocr_dataset/`

## 2. Language Model Corpus (Tamil/Arabic)
**Purpose:** Teach the NLP model (IndicBERT, BiLSTM) the rules, vocabulary, and grammar of the target language.
**Input/Target:** Clean textual corpora (e.g., Sangam literature) completely decoupled from any OCR images.
**Location:** `data/tamil_corpus/` and `data/arabic_corpus/`

## 3. Reconstruction Dataset
**Purpose:** Train/evaluate the model's ability to fix corrupted text.
**Input:** Corrupted text strings (e.g., "தமிழ் மொ_ி மிகவும் பழமையானது.").
**Target:** Clean text strings (e.g., "தமிழ் மொழி மிகவும் பழமையானது.").
**Location:** `data/reconstruction/`

## Manifests
Data is indexed via JSONL manifest files located in `data/manifests/`. Training scripts must consume these manifests rather than crawling directories.

### OCR Manifest Format
```json
{
    "id": "tamil_000001",
    "image": "data/segmented/tamil/lines/000001.png",
    "text": "தமிழ் மொழி",
    "language": "tamil",
    "script": "tamil",
    "direction": "ltr",
    "split": "train"
}
```

### Language Model Manifest Format
```json
{
    "id": "tamil_lm_000001",
    "text": "தமிழ் மொழி மிகவும் பழமையானது.",
    "language": "tamil",
    "split": "train"
}
```

### Reconstruction Manifest Format
```json
{
    "id": "recon_000001",
    "corrupted": "தமிழ் மொ_ி மிகவும் பழமையானது.",
    "target": "தமிழ் மொழி மிகவும் பழமையானது.",
    "language": "tamil",
    "corruption_type": "character"
}
```

## Splitting Strategy
Data splitting is performed deterministically via `scripts/split_dataset.py`. Splitting for OCR must occur at the document/page level to prevent data leakage (lines from the same page appearing in both train and test sets).
