import sys
import yaml
import torch
import json
from pathlib import Path

# Fix Windows console encoding for Tamil characters
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.recognition.crnn import CRNN
from models.digitalization.tokenizer import Tokenizer
from models.recognition.uncertainty import OCRUncertaintyDetector
from models.reconstruction.restoration_model import mT5RestorationModel
from models.language_understanding.tamil_language_model import TamilLanguageModel
from models.reconstruction.ranker import CandidateRanker, DecisionLayer
from src.pipeline.end_to_end import TamilTextPipeline
from scripts.dataset_loader import get_dataloader
from utils.device import get_device

def load_config():
    with open(PROJECT_ROOT / "configs" / "architecture" / "final_system.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    print("=== ARCHITECTURE PIPELINE TEST ===")
    config = load_config()
    device = get_device()
    
    print("1. Loading OCR...")
    vocab_path = PROJECT_ROOT / config["ocr"]["vocabulary"]
    ocr_tokenizer = Tokenizer(vocab_path=str(vocab_path))
    ocr_model = CRNN(ocr_tokenizer.num_classes, feature_size=2048).to(device)
    ocr_model.load_state_dict(torch.load(PROJECT_ROOT / config["ocr"]["checkpoint"], map_location=device)["model_state_dict"])
    ocr_model.eval()
    
    print("2. Loading Uncertainty Detector...")
    uncertainty_detector = OCRUncertaintyDetector()
    
    print("3. Loading mT5...")
    mt5_dir = PROJECT_ROOT / config["restoration"]["model_dir"]
    restoration_model = mT5RestorationModel(str(mt5_dir), device=device)
    
    print("4. Loading IndicBERT...")
    lm_dir = PROJECT_ROOT / config["mlm"]["model_dir"]
    language_model = TamilLanguageModel(checkpoint_dir=str(lm_dir))
    language_model.load_model()
    
    print("5. Loading Ranker and Decision Layer...")
    ranker = CandidateRanker()
    decision_layer = DecisionLayer()
    
    print("6. Initializing Pipeline...")
    pipeline = TamilTextPipeline(
        ocr_model=ocr_model,
        ocr_tokenizer=ocr_tokenizer,
        uncertainty_detector=uncertainty_detector,
        restoration_model=restoration_model,
        language_model=language_model,
        ranker=ranker,
        decision_layer=decision_layer
    )
    
    print("7. Getting Test Data...")
    working_root = PROJECT_ROOT / "data" / "tamil_ocr_dataset"
    
    val_loader = get_dataloader(
        dataset_root=str(working_root),
        language="tamil",
        split="validation",
        batch_size=1,
        tokenizer=ocr_tokenizer,
        num_workers=0
    )
    
    print("\nRunning Inference...")
    for i, batch in enumerate(val_loader):
        if i >= 5: # Just 5 examples
            break
            
        images = batch[0].to(device)
        print(f"\n--- Example {i+1} ---")
        
        # Test full pipeline
        provenance = pipeline.process(images[0])
        
        print(f"OCR TEXT: {provenance['ocr_text']}")
        print(f"OCR CONF: {provenance['ocr_sequence_confidence']:.4f}")
        print(f"UNCERTAINTY: {provenance['uncertainty_detected']} ({provenance['uncertainty_reason']})")
        
        if provenance["restoration_candidates"]:
            print(f"TOP CANDIDATE: {provenance['ranked_candidates'][0]['text']} (Score: {provenance['ranked_candidates'][0]['final_score']:.4f})")
        print(f"DECISION: {provenance['final_decision']}")
        print(f"FINAL TEXT: {provenance['final_text']}")
        
    print("\nArchitecture pipeline test complete.")

if __name__ == "__main__":
    main()
