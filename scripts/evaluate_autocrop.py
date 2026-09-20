import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import torch
import json
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from pathlib import Path
import glob

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

def get_autocrop_bbox(img_np, threshold=240, margin_x=2, margin_y=2, min_pixels=10):
    fg_mask = img_np < threshold
    y_indices, x_indices = np.where(fg_mask)
    
    if len(y_indices) < min_pixels:
        return None  # Fallback
        
    h, w = img_np.shape
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # Check if box is suspiciously small (e.g. less than 2x2)
    if (x_max - x_min) < 2 or (y_max - y_min) < 2:
        return None
        
    x_min = max(0, x_min - margin_x)
    x_max = min(w - 1, x_max + margin_x)
    y_min = max(0, y_min - margin_y)
    y_max = min(h - 1, y_max + margin_y)
    
    return x_min, y_min, x_max, y_max

def calc_fg_pct(img_np, threshold=240):
    fg_mask = img_np < threshold
    return (np.sum(fg_mask) / (img_np.shape[0] * img_np.shape[1])) * 100

def get_text_height_ratio(img_np, threshold=240):
    fg_mask = img_np < threshold
    y_indices, x_indices = np.where(fg_mask)
    if len(y_indices) == 0: return 0.0
    content_h = np.max(y_indices) - np.min(y_indices) + 1
    return content_h / max(1, img_np.shape[0])

def evaluate_image(name, img_path, model, transform, tokenizer, device, out_dir, margin_x=0, margin_y=0, threshold=240):
    if not isinstance(img_path, Image.Image):
        img = Image.open(img_path).convert('L')
    else:
        img = img_path
        
    img_np = np.array(img)
    orig_w, orig_h = img.size
    
    # Original stats
    orig_fg = calc_fg_pct(img_np, threshold)
    orig_ratio = get_text_height_ratio(img_np, threshold)
    
    # Original Inference
    pred_orig = run_inference(model, img, transform, tokenizer, device)
    
    # AutoCrop
    bbox = get_autocrop_bbox(img_np, threshold, margin_x, margin_y)
    
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))
        crop_w, crop_h = cropped.size
        
        crop_np = np.array(cropped)
        crop_fg = calc_fg_pct(crop_np, threshold)
        crop_ratio = get_text_height_ratio(crop_np, threshold)
        
        # Draw bbox on original for preview
        preview_orig = img.convert('RGB')
        draw = ImageDraw.Draw(preview_orig)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=1)
        
        # Save preview
        preview_orig.save(os.path.join(out_dir, f"{name}_1_original_bbox.png"))
        cropped.save(os.path.join(out_dir, f"{name}_2_cropped.png"))
        
        # Model input preview (after resize)
        model_in = transform(cropped)
        model_in_img = transforms.ToPILImage()((model_in + 1) / 2.0)
        model_in_img.save(os.path.join(out_dir, f"{name}_3_model_input.png"))
        
    else:
        # Fallback
        cropped = img
        crop_w, crop_h = orig_w, orig_h
        crop_fg = orig_fg
        crop_ratio = orig_ratio
        bbox = "FALLBACK"
        x_min, y_min, x_max, y_max = 0, 0, orig_w-1, orig_h-1
        
    # Cropped Inference
    pred_crop = run_inference(model, cropped, transform, tokenizer, device)
    
    print(f"\n--- {name} ---")
    print(f"Original Size      : {orig_w}x{orig_h}")
    print(f"Crop Box           : {bbox}")
    print(f"Cropped Size       : {crop_w}x{crop_h}")
    print(f"Orig FG% / Ratio   : {orig_fg:.2f}% / {orig_ratio:.4f}")
    print(f"Crop FG% / Ratio   : {crop_fg:.2f}% / {crop_ratio:.4f}")
    print(f"Baseline Pred      : {pred_orig}")
    print(f"AutoCrop Pred      : {pred_crop}")
    
    if pred_orig != pred_crop:
        print(">>> PREDICTION CHANGED <<<")
    else:
        print(">>> PREDICTION REMAINED IDENTICAL <<<")

    return {
        "name": name,
        "orig_size": f"{orig_w}x{orig_h}",
        "crop_box": str(bbox),
        "crop_size": f"{crop_w}x{crop_h}",
        "pred_orig": pred_orig,
        "pred_crop": pred_crop
    }

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

    out_dir = "outputs/training/tamil/refinement/exp_autocrop_inference/previews"
    
    print("==================================================")
    print("3. & 4. TEST IMAGES")
    print("==================================================")
    
    results = []
    
    if os.path.exists("sample_test.png"):
        results.append(evaluate_image("sample_test", "sample_test.png", model, transform, tokenizer, device, out_dir))
        
    if os.path.exists("sample.jpg"):
        results.append(evaluate_image("sample", "sample.jpg", model, transform, tokenizer, device, out_dir))

    print("\n==================================================")
    print("8. ROBUSTNESS TESTS")
    print("==================================================")
    
    # Get a few training samples
    manifest_path = "data/tamil_ocr_dataset/imported/tamil/train/packet_001/manifest.jsonl"
    img_dir = "data/tamil_ocr_dataset/imported/tamil/train/packet_001"
    
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
            if len(samples) >= 3: break
            
    # A. Tightly cropped training image
    t_img_path = os.path.join(img_dir, samples[0]["image"])
    results.append(evaluate_image("robust_A_tight", t_img_path, model, transform, tokenizer, device, out_dir))
    
    # B. Padded image
    img = Image.open(t_img_path).convert('L')
    w, h = img.size
    canvas = Image.new('L', (w*2, h*2), color=255)
    canvas.paste(img, (w//2, h//2))
    results.append(evaluate_image("robust_B_padded", canvas, model, transform, tokenizer, device, out_dir))
    
    # C. Image with small margins (10px)
    canvas_small = Image.new('L', (w+20, h+20), color=255)
    canvas_small.paste(img, (10, 10))
    results.append(evaluate_image("robust_C_small_margin", canvas_small, model, transform, tokenizer, device, out_dir))

    # Output markdown table for the walkthrough
    print("\n=== MARKDOWN TABLE ===\n")
    print("| Image | Original Size | Crop Box | Cropped Size | Baseline Prediction | AutoCrop Prediction |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        print(f"| {r['name']} | {r['orig_size']} | {r['crop_box']} | {r['crop_size']} | {r['pred_orig']} | {r['pred_crop']} |")

if __name__ == "__main__":
    main()
