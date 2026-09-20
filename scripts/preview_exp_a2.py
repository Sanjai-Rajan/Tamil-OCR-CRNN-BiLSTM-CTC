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

def degrade_image(img, scale):
    w, h = img.size
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    down_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    up_img = down_img.resize((w, h), Image.Resampling.LANCZOS)
    return up_img

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

    out_dir = "outputs/training/tamil/refinement/exp_A2_resolution/previews"
    
    manifest_path = "data/tamil_ocr_dataset/imported/tamil/train/packet_001/manifest.jsonl"
    img_dir = "data/tamil_ocr_dataset/imported/tamil/train/packet_001"
    
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
            if len(samples) >= 20: break
            
    scales = [1.0, 0.75, 0.50, 0.375, 0.25, 0.20]
    scale_names = ["100pct", "75pct", "50pct", "37.5pct", "25pct", "20pct"]
    
    diagnostic_results = []
    
    for i, data in enumerate(samples):
        img_path = os.path.join(img_dir, data['image'])
        img = Image.open(img_path).convert('L')
        
        for scale, name in zip(scales, scale_names):
            if scale == 1.0:
                variant = img
            else:
                variant = degrade_image(img, scale)
                
            variant.save(os.path.join(out_dir, f"sample_{i:02d}_{name}.png"))
            
            if i < 3:  # Only do inference for the first 3 samples
                pred = run_inference(model, variant, transform, tokenizer, device)
                diagnostic_results.append({
                    "sample": i,
                    "gt": data["label"],
                    "scale": name,
                    "pred": pred
                })

    print("==================================================")
    print("OFFLINE DIAGNOSTIC PREDICTIONS")
    print("==================================================")
    
    for i in range(3):
        print(f"\nSample {i}:")
        for res in diagnostic_results:
            if res["sample"] == i:
                print(f"{res['scale']:>10}: {res['pred']}")

if __name__ == "__main__":
    main()
