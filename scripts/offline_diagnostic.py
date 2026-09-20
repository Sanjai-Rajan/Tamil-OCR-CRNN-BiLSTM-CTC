import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import torch
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from models.recognition.crnn import CRNN
from models.digitalization.tokenizer import Tokenizer
from models.recognition.decoder import ctc_decode
from scripts.dataset_loader import AspectRatioPreservingResize

def run_inference(model, image, transform, tokenizer, device):
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        preds = ctc_decode(outputs)
        pred_seq = preds[0].cpu().tolist()
        clean_pred = []
        prev = -1
        for p in pred_seq:
            if p != 0 and p != prev:
                clean_pred.append(p)
            prev = p
        try:
            pred_text = tokenizer.decode(clean_pred)
        except AttributeError:
            pred_text = "".join([str(c) for c in clean_pred])
    return pred_text

def get_content_bbox(img_np, threshold=240):
    fg_mask = img_np < threshold
    y_indices, x_indices = np.where(fg_mask)
    if len(y_indices) == 0:
        return 0, 0, 0, 0
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return x_min, y_min, x_max, y_max

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_path = "data/tamil_ocr_dataset/vocabulary/tamil_vocab.json"
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    baseline_ckpt = "checkpoints/recognition/tamil/tamil_full_40epoch/best.pth"
    model = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    checkpoint = torch.load(baseline_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = transforms.Compose([
        AspectRatioPreservingResize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    manifest_path = "data/tamil_ocr_dataset/imported/tamil/train/packet_001/manifest.jsonl"
    img_dir = "data/tamil_ocr_dataset/imported/tamil/train/packet_001"
    
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            samples.append(data)
            if len(samples) >= 3:
                break
                
    print("==================================================")
    print("PHASE 7 - OFFLINE DIAGNOSTIC")
    print("==================================================")
    
    for i, data in enumerate(samples):
        img_path = os.path.join(img_dir, data['image'])
        img = Image.open(img_path).convert('L')
        w, h = img.size
        
        # 1. Original
        pred_orig = run_inference(model, img, transform, tokenizer, device)
        
        # 2. Simulated sample_test.png (Large white canvas, text is 16% of height)
        # Pad image so that original height is 16% of new height
        new_h = int(h / 0.16)
        new_w = int(w / 0.16)
        canvas = Image.new('L', (new_w, new_h), color=255)
        offset_x = (new_w - w) // 2
        offset_y = (new_h - h) // 2
        canvas.paste(img, (offset_x, offset_y))
        pred_padded = run_inference(model, canvas, transform, tokenizer, device)
        
        # 3. Tightly framed variant (Crop the canvas back to the text)
        canvas_np = np.array(canvas)
        x_min, y_min, x_max, y_max = get_content_bbox(canvas_np)
        cropped_canvas = canvas.crop((x_min, y_min, x_max+1, y_max+1))
        pred_cropped = run_inference(model, cropped_canvas, transform, tokenizer, device)
        
        # 4. Final model input representation (Low res but tightly framed)
        # To simulate the resolution degradation WITHOUT padding, we scale down to 16% and scale back up.
        small_w, small_h = max(1, int(w * 0.16)), max(1, int(h * 0.16))
        degraded = img.resize((small_w, small_h), Image.Resampling.LANCZOS)
        pred_degraded = run_inference(model, degraded, transform, tokenizer, device)
        
        print(f"\nSample {i+1}: {data['label']}")
        print("-" * 50)
        print(f"1. Original                 : {pred_orig}")
        print(f"2. Padded (Simulate sample) : {pred_padded}")
        print(f"3. Padded -> Re-cropped     : {pred_cropped}")
        print(f"4. Degraded Res (Tight)     : {pred_degraded}")

if __name__ == "__main__":
    main()
