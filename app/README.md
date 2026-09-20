# Tamil Manuscript OCR Web Application

This directory contains the local web frontend and FastAPI backend for the Tamil Manuscript OCR project.

## Prerequisites
Ensure you are in the project's virtual environment (if applicable) and have installed all requirements from the root project, plus `fastapi`, `uvicorn`, and `python-multipart`.

```bash
pip install fastapi uvicorn python-multipart
```

## How to Start the Backend
From the root of the project (`1_Draft`), run the following command to start the FastAPI server:

```bash
python -m uvicorn app.main:app --reload
```

## How to Open the Frontend
Once the server is running, open your web browser and navigate to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

## Model Used
- **Architecture**: CRNN (Convolutional Recurrent Neural Network)
- **Feature Size**: 2048
- **Decoding**: CTC (Connectionist Temporal Classification)

## Checkpoint Used
- **Path**: `checkpoints/recognition/tamil/tamil_full_40epoch/best.pth`
- The model is loaded automatically on backend startup.

## Example Request
The API expects a `multipart/form-data` POST request to `/api/ocr` with an image file.

```bash
curl -X POST "http://127.0.0.1:8000/api/ocr" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample.jpg"
```

## Example Response
```json
{
  "success": true,
  "text": "அம்மா",
  "processing_time": 0.0452,
  "device": "cuda",
  "checkpoint": "best.pth",
  "confidence": 0.9854
}
```

## Troubleshooting
- **Model Loading Failed**: Check if the `best.pth` checkpoint exists in the expected directory. Also verify the `tamil_vocab.json` path.
- **Out of Memory**: The model uses the GPU if available. If you encounter CUDA OOM errors, ensure no other heavy processes are using the GPU, or force PyTorch to use the CPU by setting environment variables (`set CUDA_VISIBLE_DEVICES=""` on Windows).
