import sys
import os
import time
import requests
import numpy as np
import cv2
from PIL import Image, ImageOps
from pathlib import Path
import json
import torch

sys.stdout.reconfigure(encoding='utf-8')
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from models.cv.segmenter import Segmenter
from app.inference import OCRService

def create_10line_image():
    # Take 10 images from dataset and stack them
    img_dir = root_dir / "data" / "tamil_ocr_dataset" / "imported" / "tamil" / "train" / "packet_014" / "images"
    images = list(img_dir.glob("*.jpg"))[:10]
    
    loaded_imgs = []
    max_w = 0
    for p in images:
        img = Image.open(p).convert('L')
        # add some padding
        img = ImageOps.expand(img, border=10, fill=255)
        loaded_imgs.append(img)
        if img.width > max_w: max_w = img.width

    # Create a white canvas
    total_h = sum(img.height for img in loaded_imgs) + 50
    canvas = Image.new('L', (max_w + 20, total_h), 255)
    
    y_offset = 20
    for img in loaded_imgs:
        canvas.paste(img, (10, y_offset))
        y_offset += img.height

    out_path = root_dir / "sample_10lines.png"
    canvas.save(out_path)
    return out_path

def main():
    img_path = create_10line_image()
    print(f"Created 10-line test image at {img_path}")
    
    img = Image.open(img_path)
    print(f"\n=== STEP 1: VERIFY SEGMENTATION DIRECTLY ===")
    print(f"Input image dimensions: {img.width}x{img.height}")
    
    segmenter = Segmenter()
    # Segmenter accepts PIL image and returns PIL crops
    crops = segmenter.segment_lines(img)
    print(f"Detected line count: {len(crops)}")
    
    print("\n=== STEP 2: SAVE EVERY LINE CROP ===")
    debug_dir = root_dir / "outputs" / "fast_track" / "debug_lines"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    for i, crop in enumerate(crops):
        crop_path = debug_dir / f"line_{i+1:02d}.png"
        crop.save(crop_path)
        print(f"Line {i+1}: width={crop.width}, height={crop.height} saved to {crop_path.name}")
        
    print("\n=== STEP 3: CREATE A DEBUG CONTACT SHEET ===")
    total_h = sum(c.height for c in crops) + len(crops)*10
    max_w = max(c.width for c in crops)
    contact = Image.new('L', (max_w + 50, total_h), 255)
    y_offset = 0
    for i, c in enumerate(crops):
        contact.paste(c, (50, y_offset))
        y_offset += c.height + 10
    contact.save(debug_dir / "segmentation_debug.png")
    print("Saved segmentation_debug.png")
    
    print("\n=== STEP 4 & 5: RECOGNIZE EACH LINE DIRECTLY ===")
    ocr = OCRService()
    
    recognized_lines = []
    
    for i, crop in enumerate(crops):
        start_time = time.time()
        # manual processing similar to predict() but single image to get stats
        img_l = crop.convert('L')
        img_tensor = ocr.transform(img_l).unsqueeze(0).to(ocr.device)
        with torch.no_grad():
            outputs = ocr.model(img_tensor)
            from models.recognition.decoder import ctc_decode_with_confidence
            decoded_indices, confidences = ctc_decode_with_confidence(outputs)
            
        pred_seq = decoded_indices[0].cpu().tolist()
        char_confs = confidences[0]
        
        clean_pred = []
        clean_confs = []
        prev = -1
        for p, conf in zip(pred_seq, char_confs):
            if p != 0 and p != prev:
                clean_pred.append(p)
                clean_confs.append(conf)
            prev = p
            
        text = ocr.tokenizer.decode(clean_pred)
        recognized_lines.append(text)
        
        proc_time = time.time() - start_time
        conf_val = sum(clean_confs)/len(clean_confs) if clean_confs else 0.0
        
        print(f"Line {i+1}:")
        print(f"Prediction: {text}")
        print(f"Confidence: {conf_val:.4f}")
        print(f"Length: {len(text)}")
        print(f"Processing time: {proc_time:.4f}s\n")
        
    print("=== STEP 6: VERIFY AGGREGATION ===")
    print(f"Detected lines: {len(crops)}")
    print(f"Recognized lines: {len(recognized_lines)}")
    final_text = '\\n'.join(recognized_lines)
    print(f"Returned lines: {len(final_text.split('\\n'))}")
    print(f"Final returned text:\\n{final_text}")
    
    print("\n=== STEP 7: FRONTEND API RESPONSE ===")
    url = "http://127.0.0.1:8000/api/ocr"
    try:
        with open(img_path, "rb") as f:
            resp = requests.post(url, files={"file": ("sample_10lines.png", f, "image/png")})
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"API Error: {e}")

if __name__ == "__main__":
    main()
