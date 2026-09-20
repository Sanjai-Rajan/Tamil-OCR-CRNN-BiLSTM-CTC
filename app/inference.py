import sys
import os
from pathlib import Path
import time
import torch
from torchvision import transforms
from PIL import Image

# Add root directory to sys.path to allow imports from models and src
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode_with_confidence
from models.digitalization.tokenizer import Tokenizer
from utils.device import get_device
from models.cv.segmenter import Segmenter
from models.digitalization.conceptual_corrector import DeterministicCorrector

from PIL import ImageFilter

class AspectRatioPreservingResize:
    def __init__(self, height=32, apply_unsharp=True):
        self.height = height
        self.apply_unsharp = apply_unsharp
        
    def __call__(self, img):
        if self.apply_unsharp:
            # Mild unsharp masking to preserve thin strokes before aggressive downsampling
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
        w, h = img.size
        new_w = max(4, round(w * self.height / h))
        return img.resize((new_w, self.height), Image.Resampling.LANCZOS)

class OCRService:
    def __init__(self):
        self.device = get_device()
        self.vocab_path = root_dir / "data" / "tamil_ocr_dataset" / "vocabulary" / "tamil_vocab.json"
        self.checkpoint_path = root_dir / "checkpoints" / "recognition" / "tamil" / "ocr_balanced_temporal" / "best.pth"
        
        self.tokenizer = None
        self.model = None
        self.transform = None
        self.model_loaded = False
        self.segmenter = Segmenter()
        self.corrector = DeterministicCorrector()
        self.load_model()

    def load_model(self):
        try:
            print(f"[OCRService] Initializing Tokenizer from {self.vocab_path}")
            self.tokenizer = Tokenizer(vocab_path=str(self.vocab_path))
            
            print(f"[OCRService] Initializing CRNN on {self.device}")
            self.model = CRNN(self.tokenizer.num_classes, feature_size=2048, less_downsample=True).to(self.device)
            
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")
                
            print(f"[OCRService] Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            
            self.transform = transforms.Compose([
                AspectRatioPreservingResize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            self.model_loaded = True
            
            print("[OCRService] Model loaded successfully.")
            if self.device.type == 'cuda':
                print(f"[OCRService] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[OCRService] Vocabulary size: {self.tokenizer.num_classes}")
            
        except Exception as e:
            print(f"[OCRService] ERROR loading model: {e}")
            self.model_loaded = False
            raise e

    def predict(self, image: Image.Image) -> dict:
        if not self.model_loaded:
            raise RuntimeError("Model is not loaded.")
            
        start_time = time.time()
        
        # 1. Segment Image into lines
        lines = self.segmenter.segment_lines(image)
        
        recognized_lines = []
        raw_lines = []
        all_confs = []
        words_detected = 0
        self.corrections_made = 0
        self.corrections_list = []
        
        for line_img in lines:
            # 2. Segment line into words
            words = self.segmenter.segment_words(line_img)
            words_detected += len(words)
            
            line_text_parts = []
            raw_text_parts = []
            
            for word_data in words:
                word_img = word_data['image']
                word_conf = word_data['confidence']
                
                # 3. Preprocess word
                img = word_img.convert('L')
                img_tensor = self.transform(img)
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                
                # 4. Inference
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    decoded_indices, confidences = ctc_decode_with_confidence(outputs)
                    
                pred_seq = decoded_indices[0].cpu().tolist()
                char_confs = confidences[0]
                
                # 5. Clean up CTC output
                clean_pred = []
                clean_confs = []
                prev = -1
                for p, conf in zip(pred_seq, char_confs):
                    if p != 0 and p != prev:
                        clean_pred.append(p)
                        clean_confs.append(conf)
                    prev = p
                    
                # 6. Decode word to text
                recognized_word = self.tokenizer.decode(clean_pred)
                raw_text_parts.append(recognized_word)
                
                # 7. Apply deterministic conceptual correction
                corrected_word, correction_status = self.corrector.correct(recognized_word)
                if correction_status == 'corrected':
                    self.corrections_made += 1
                    self.corrections_list.append({
                        "raw": recognized_word,
                        "corrected": corrected_word
                    })
                    
                line_text_parts.append(corrected_word)
                
                # Combine OCR character confidence with segmentation confidence
                if clean_confs:
                    avg_char_conf = sum(clean_confs) / len(clean_confs)
                else:
                    avg_char_conf = 0.0
                all_confs.append(avg_char_conf * word_conf)
                
            # Reconstruct line text
            recognized_lines.append(" ".join(line_text_parts))
            raw_lines.append(" ".join(raw_text_parts))
            
        final_text = "\n".join(recognized_lines)
        raw_final_text = "\n".join(raw_lines)
        processing_time = time.time() - start_time
        
        sequence_confidence = 0.0
        if all_confs:
            sequence_confidence = sum(all_confs) / len(all_confs)
            
        return {
            "success": True,
            "text": final_text,
            "raw_text": raw_final_text,
            "processing_time": round(processing_time, 4),
            "device": str(self.device),
            "checkpoint": self.checkpoint_path.name,
            "confidence": round(sequence_confidence, 4),
            "lines_detected": len(lines),
            "words_detected": words_detected,
            "corrections_made": self.corrections_made,
            "corrections_list": self.corrections_list,
            "cer": None,
            "wer": None,
            "character_accuracy": None,
            "word_accuracy": None,
            "blank_percentage": None
        }
