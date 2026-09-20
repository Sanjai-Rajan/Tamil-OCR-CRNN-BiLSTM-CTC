# Final Production AI / Pretrained Model Forensic Audit

2026-09-19T16:31:00+05:30

## Active OCR Architecture
The active production Tamil OCR architecture strictly uses an in-house developed CNN-RNN structure:
- Custom ResNet-based CNN Feature Extractor
- BiLSTM Sequence Modeling (2 layers, 256 hidden size)
- Final Linear Projection Head predicting CTC log-probabilities
- Custom Deterministic Conceptual Corrector

There is **NO** reliance on, or loading of, any external or pretrained Transformer, BERT, mT5, or large language models. The architecture is entirely self-contained.

## Active Dependencies
The production system is verified to only require:
- PyTorch (torch, torchvision, torchaudio)
- OpenCV (opencv-python)
- Pillow
- NumPy, Pandas, scikit-learn
- FastAPI, Uvicorn, Streamlit
- Albumentations, tqdm

## Prohibited AI Search
Findings from the active codebase (`app/`, `models/`, `scripts/` excluding `historical/`):
- `mT5`, `T5`: NOT FOUND (except inside test/diagnostic scripts `test_architecture.py` which are inactive).
- `BERT`, `IndicBERT`, `RoBERTa`: NOT FOUND.
- `GPT`, `LLM`: NOT FOUND.
- `transformers`, `huggingface`, `huggingface_hub`, `safetensors`: NOT FOUND in production code.
- `pretrained`, `from_pretrained`: HISTORICAL (Found in `scripts\data\prepare_tamil_lm_corpus.py` which was used for NLP data prep).
- **External AI APIs** (`openai`, `anthropic`, `gemini`): NOT FOUND. Generic HTTP clients (`requests`) are used ONLY for fetching sample images or for localhost API testing (`debug_multiline.py`, `test_frontend.py`).

## Historical Artifacts
The following scripts contain old AI/MLM references:
- `mlm_smoke_test.py`
- `mlm_checkpoint_test.py`
- `prepare_tamil_lm_corpus.py`
- `monitor_mlm_live.py`
- `test_architecture.py`

**These are archived historical artifacts and are not imported by the active production OCR pipeline.**

## Model Weight Audit
An exhaustive search for `*.pth`, `*.pt`, `*.bin`, `*.safetensors`, and `*.ckpt` outside the `venv/` returned ONLY standard PyTorch `.pth` files residing safely within `checkpoints\recognition\tamil\`.
- All discovered weights are exact ~124 MB PyTorch checkpoints from CRNN training epochs.
- ZERO undocumented pretrained weights or safetensors blobs exist in the project directory.

## External API Audit
No external AI API calls were found in the active source. Localhost tests utilize `requests.post()` but do not connect to external inference providers.

## Production Verification
- **OCRService import**: OK
- **FastAPI import**: OK
- **Checkpoint existence**: Verified
- **Checkpoint SHA-256**: `07554040440E80E5EC2B1BD6EA1FA5BB6618E105AE3AD45F7590F21113E2F136`
- **Requirements status**: Verified clean.

## Safety Result
**NO PROHIBITED PRETRAINED MODEL OR EXTERNAL AI IS ACTIVELY USED BY PRODUCTION.** The environment is strictly clean.

## Files Deleted
ZERO source files were deleted. Only generated Python cache files (`__pycache__`, `*.pyc`) and obsolete `sentencepiece` were removed.
