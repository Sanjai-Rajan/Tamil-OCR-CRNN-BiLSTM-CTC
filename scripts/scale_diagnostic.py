import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import torch
import json
from pathlib import Path
from PIL import Image
import numpy as np

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode
from models.digitalization.tokenizer import Tokenizer
from scripts.dataset_loader import AspectRatioPreservingResize
from torchvision import transforms

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
    return pred_text, input_tensor.shape

def main():
    print("==================================================")
    print("STEP 1 - PREPARING DIAGNOSTIC")
    print("==================================================")
    
    # Load manifest to find an image and its ground truth
    manifest_path = "data/tamil_ocr_dataset/imported/tamil/train/packet_001/manifest.jsonl"
    print("Finding a correctly recognized training image...")
    target_data = None
    orig_img = None
    ground_truth = ""
    orig_w, orig_h = 0, 0
    
    # Initialize model first
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_path = "data/tamil_ocr_dataset/vocabulary/tamil_vocab.json"
    tokenizer = Tokenizer(vocab_path=vocab_path)
    model = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    
    ckpt_path = "checkpoints/recognition/tamil/tamil_full_40epoch/best.pth"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.Compose([
        AspectRatioPreservingResize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            img_path = os.path.join("data/tamil_ocr_dataset/imported/tamil/train/packet_001", data["image"])
            try:
                img = Image.open(img_path).convert('L')
                pred, _ = run_inference(model, img, transform, tokenizer, device)
                if pred == data["label"]:
                    target_data = data
                    orig_img = img
                    ground_truth = data["label"]
                    orig_w, orig_h = img.size
                    break
            except Exception as e:
                pass
                
    if not target_data:
        print("Could not find a perfectly recognized image in the first packet.")
        sys.exit(1)
        
    print(f"Selected Image: {img_path}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Original Dimensions: {orig_w}x{orig_h}")
    
    out_dir = Path("outputs/training/tamil/refinement/scale_diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    scales = [
        ("Original", 1.0),
        ("75%", 0.75),
        ("50%", 0.50),
        ("37.5%", 0.375),
        ("25%", 0.25)
    ]
    
    results = []
    
    print("\n==================================================")
    print("STEP 2 - RUNNING SCALE VARIANTS")
    print("==================================================")
    
    for scale_name, scale_factor in scales:
        new_w = max(1, int(orig_w * scale_factor))
        new_h = max(1, int(orig_h * scale_factor))
        
        # Downscale text
        shrunk_img = orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create a white canvas of the ORIGINAL size
        canvas = Image.new('L', (orig_w, orig_h), color=255)
        
        # Paste the shrunk image in the center (or top-left, doesn't matter much)
        offset_x = (orig_w - new_w) // 2
        offset_y = (orig_h - new_h) // 2
        canvas.paste(shrunk_img, (offset_x, offset_y))
        
        # Run standard inference pipeline on the padded canvas
        pred, input_shape = run_inference(model, canvas, transform, tokenizer, device)
        
        # Save visualization
        canvas.save(out_dir / f"variant_{scale_name.replace('%', '')}.png")
        
        # Also save the model-input representation
        input_tensor = transform(canvas)
        # un-normalize for saving
        input_tensor = (input_tensor * 0.5) + 0.5
        input_pil = transforms.ToPILImage()(input_tensor)
        input_pil.save(out_dir / f"model_input_{scale_name.replace('%', '')}.png")
        
        is_correct = "YES" if pred == ground_truth else "NO"
        
        results.append({
            'scale': scale_name,
            'dimensions': f"{new_w}x{new_h}",
            'effective_h': f"{new_h}px",
            'input_shape': f"1x32x{input_shape[3]}",
            'prediction': pred,
            'correct': is_correct
        })
        
    print("\n==================================================")
    print("FINAL REPORT - RESULT TABLE")
    print("==================================================")
    print(f"Ground Truth: {ground_truth}")
    print("-" * 110)
    print(f"{'Scale':<10} | {'Dimensions':<15} | {'Effective H':<12} | {'Input Shape':<15} | {'Prediction':<30} | {'Correct?'}")
    print("-" * 110)
    for res in results:
        print(f"{res['scale']:<10} | {res['dimensions']:<15} | {res['effective_h']:<12} | {res['input_shape']:<15} | {res['prediction']:<30} | {res['correct']}")
        
if __name__ == '__main__':
    main()
