import sys
import os
import torch
import cv2
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Set up paths
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode
from models.digitalization.tokenizer import Tokenizer
from utils.device import get_device

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img

def preprocess_for_crnn(img):
    # CRNN expects (1, 32, W)
    h, w = img.shape
    new_h = 32
    new_w = int(w * (new_h / h))
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Normalize
    img_normalized = img_resized.astype("float32") / 255.0
    img_normalized = (img_normalized - 0.5) / 0.5
    
    # To tensor
    tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0) # (1, 1, 32, W)
    return tensor

def segment_lines(img):
    # Simple line segmentation using horizontal projection profile
    # Inverse binary
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Horizontal projection
    proj = torch.tensor(thresh).sum(dim=1)
    
    # Find lines
    lines = []
    in_line = False
    start = 0
    for i, val in enumerate(proj):
        if val > 0 and not in_line:
            in_line = True
            start = i
        elif val == 0 and in_line:
            in_line = False
            # Ensure minimum height
            if i - start > 5:
                lines.append((start, i))
                
    if in_line and len(proj) - start > 5:
         lines.append((start, len(proj)))
         
    # Extract images
    line_imgs = []
    for (start, end) in lines:
        # Add slight padding
        pad = 2
        s = max(0, start - pad)
        e = min(img.shape[0], end + pad)
        line_imgs.append(img[s:e, :])
        
    return line_imgs

def run_inference():
    device = get_device()
    vocab_path = "data/tamil_ocr_dataset/vocabulary/tamil_vocab.json"
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    model = CRNN(tokenizer.num_classes, feature_size=2048, less_downsample=True).to(device)
    ckpt_path = "checkpoints/recognition/tamil/ocr_temporal_fix/best.pth"
    print(f"Loading {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    test_files = ["sample.jpg", "sample_test.png", "sample_10lines.png"]
    
    print("\n============================================================")
    print("REAL IMAGE TEST")
    print("============================================================")
    
    with torch.no_grad():
        for filename in test_files:
            if not os.path.exists(filename):
                print(f"File {filename} not found.")
                continue
                
            print(f"\nProcessing {filename}...")
            img = load_image(filename)
            
            if filename == "sample_10lines.png":
                # Line segmentation
                lines = segment_lines(img)
                print(f"Segmented into {len(lines)} lines.")
                full_text = []
                for idx, line_img in enumerate(lines):
                    tensor = preprocess_for_crnn(line_img).to(device)
                    outputs = model(tensor)
                    preds = ctc_decode(outputs)
                    
                    pred_seq = preds[0].cpu().tolist()
                    clean_pred = []
                    prev = -1
                    for p in pred_seq:
                        if p != 0 and p != prev:
                            clean_pred.append(p)
                        prev = p
                    
                    try:
                        text = tokenizer.decode(clean_pred)
                    except:
                        text = "".join([str(c) for c in clean_pred])
                    
                    print(f"Line {idx+1}: {text}")
                    full_text.append(text)
                print("\nReconstructed Text:")
                print("\n".join(full_text))
            else:
                # Single line assumption
                tensor = preprocess_for_crnn(img).to(device)
                outputs = model(tensor)
                preds = ctc_decode(outputs)
                
                pred_seq = preds[0].cpu().tolist()
                clean_pred = []
                prev = -1
                for p in pred_seq:
                    if p != 0 and p != prev:
                        clean_pred.append(p)
                    prev = p
                
                try:
                    text = tokenizer.decode(clean_pred)
                except:
                    text = "".join([str(c) for c in clean_pred])
                    
                print(f"Prediction: {text}")

if __name__ == "__main__":
    run_inference()
