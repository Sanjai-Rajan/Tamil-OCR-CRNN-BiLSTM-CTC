import cv2
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.recognition.crnn import CRNN
from models.cv.preprocessor import ImagePreprocessor
from utils.device import get_device
from utils.logger import get_logger

logger = get_logger("inference", "logs/inference.log")

def run_inference(image_path):
    device = get_device()
    logger.info(f"Using device: {device}")

    model = CRNN(100).to(device)
    model.eval()

    logger.info(f"Loading image from {image_path}")
    image = cv2.imread(image_path, 0)
    
    preprocessor = ImagePreprocessor()
    try:
        image = preprocessor.preprocess(image)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return

    # Convert to float32 and normalize
    tensor_image = torch.tensor(image, dtype=torch.float32) / 255.0
    tensor_image = tensor_image.unsqueeze(0).unsqueeze(0).to(device)

    logger.info(f"Running model inference with tensor shape {tensor_image.shape}")
    with torch.no_grad():
        prediction = model(tensor_image)
    
    logger.info(f"Prediction complete. Output shape: {prediction.shape}")
    
    # Mocking decode step to simulate OCR output with corruption
    # In reality, this would use models.recognition.decoder.ctc_decode
    mock_ocr_output = "அவன் அரண்ம_னை சென்றான்"
    logger.info(f"Decoded OCR output: {mock_ocr_output}")
    
    # Contextual Reconstruction
    from models.reconstruction.reconstructor import TamilReconstructor
    reconstructor = TamilReconstructor(confidence_threshold=0.8)
    
    logger.info("Running contextual reconstruction...")
    reconstruction_result = reconstructor.reconstruct(mock_ocr_output, full_paragraph=mock_ocr_output)
    
    logger.info(f"Reconstruction result: {reconstruction_result}")
    
    return {
        "prediction_tensor": prediction,
        "reconstruction": reconstruction_result
    }

if __name__ == "__main__":
    run_inference("sample.jpg")
