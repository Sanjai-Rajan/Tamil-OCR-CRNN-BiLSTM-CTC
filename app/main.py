from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import io
from PIL import Image

from app.inference import OCRService

app = FastAPI(title="Tamil OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global OCR service
ocr_service = None

@app.on_event("startup")
async def startup_event():
    global ocr_service
    print("Starting up Tamil OCR Service...")
    ocr_service = OCRService()

# Create directories if they don't exist
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = templates_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/health")
async def health_check():
    if not ocr_service:
        return {"status": "error", "message": "OCR service not initialized"}
    
    return {
        "status": "ok",
        "model_loaded": ocr_service.model_loaded,
        "device": str(ocr_service.device),
        "checkpoint": ocr_service.checkpoint_path.name if ocr_service.model_loaded else None
    }

@app.post("/api/ocr")
async def predict_ocr(file: UploadFile = File(...)):
    if not ocr_service or not ocr_service.model_loaded:
        raise HTTPException(status_code=503, detail="OCR Model is not loaded or available.")
        
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Run inference
        result = ocr_service.predict(image)
        return result
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during OCR processing.")
