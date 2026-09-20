import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import torch
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import Levenshtein

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from models.recognition.crnn import CRNN
from models.digitalization.tokenizer import Tokenizer
from models.recognition.decoder import ctc_decode
from scripts.dataset_loader import AspectRatioPreservingResize

def run_inference(model, image, transform, tokenizer, device):
    input_tensor = transform(image).unsqueeze(0).to(device)
    # Check final tensor dimensions
    _, _, h_tensor, w_tensor = input_tensor.shape
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
    return pred_text, (w_tensor, h_tensor)

def get_autocrop_bbox(img_np, threshold=240):
    fg_mask = img_np < threshold
    y_indices, x_indices = np.where(fg_mask)
    if len(y_indices) < 2: return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return x_min, y_min, x_max, y_max

def create_scale_variant(tight_img, target_h):
    w, h = tight_img.size
    new_w = max(1, int(w * (target_h / h)))
    resized = tight_img.resize((new_w, target_h), Image.Resampling.LANCZOS)
    
    canvas_h = 32
    canvas = Image.new('L', (new_w, canvas_h), color=255)
    offset_y = (canvas_h - target_h) // 2
    
    # If target_h > 32, offset_y will be negative, meaning it gets cropped by the canvas
    canvas.paste(resized, (0, offset_y))
    return canvas

def eval_variant(name, img, ground_truth, model, transform, tokenizer, device, out_dir):
    pred_text, tensor_shape = run_inference(model, img, transform, tokenizer, device)
    img.save(os.path.join(out_dir, f"{name}.png"))
    
    correct = "YES" if pred_text == ground_truth else "NO"
    dist = Levenshtein.distance(ground_truth, pred_text)
    cer = dist / max(len(ground_truth), 1)
    
    return {
        "name": name,
        "size": f"{img.size[0]}x{img.size[1]}",
        "tensor": f"{tensor_shape[0]}x{tensor_shape[1]}",
        "pred": pred_text,
        "correct": correct,
        "cer": cer
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

    # The existing preprocessing pipeline
    transform = transforms.Compose([
        AspectRatioPreservingResize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    out_dir = "outputs/training/tamil/refinement/exp_P1_scale_response/previews"
    
    print("==================================================")
    print("1. TEST IMAGE (sample_test.png)")
    print("==================================================")
    
    sample_path = "sample_test.png"
    sample_gt = "காலன்"
    img_orig = Image.open(sample_path).convert('L')
    print(f"Original size: {img_orig.size[0]}x{img_orig.size[1]}")
    
    bbox = get_autocrop_bbox(np.array(img_orig))
    print(f"Detected bounding box: {bbox}")
    
    tight_img = img_orig.crop((bbox[0], bbox[1], bbox[2] + 1, bbox[3] + 1))
    print(f"Tight crop size: {tight_img.size[0]}x{tight_img.size[1]}")
    
    heights = [4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32]
    sample_results = []
    
    # Add controls
    sample_results.append(eval_variant("sample_test_control_original", img_orig, sample_gt, model, transform, tokenizer, device, out_dir))
    sample_results.append(eval_variant("sample_test_control_autocrop", tight_img, sample_gt, model, transform, tokenizer, device, out_dir))
    
    for h in heights:
        variant = create_scale_variant(tight_img, h)
        name = f"sample_test_h{h:02d}"
        res = eval_variant(name, variant, sample_gt, model, transform, tokenizer, device, out_dir)
        res["eff_h"] = h
        sample_results.append(res)
        
    print("\n==================================================")
    print("8. SECONDARY TEST — EXISTING TRAINING IMAGES")
    print("==================================================")
    
    manifest_path = "data/tamil_ocr_dataset/imported/tamil/train/packet_001/manifest.jsonl"
    img_dir = "data/tamil_ocr_dataset/imported/tamil/train/packet_001"
    
    train_samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            train_samples.append(json.loads(line.strip()))
            if len(train_samples) >= 3: break
            
    train_results = []
    percents = [25, 40, 50, 60, 75, 100, 125]
    
    for i, t_data in enumerate(train_samples):
        t_img = Image.open(os.path.join(img_dir, t_data["image"])).convert('L')
        # Training images are already tightly cropped, but crop just to be strictly fair
        t_bbox = get_autocrop_bbox(np.array(t_img))
        if t_bbox:
            t_tight = t_img.crop((t_bbox[0], t_bbox[1], t_bbox[2] + 1, t_bbox[3] + 1))
        else:
            t_tight = t_img
            
        t_gt = t_data["label"]
        
        train_results.append(eval_variant(f"train_{i}_control", t_tight, t_gt, model, transform, tokenizer, device, out_dir))
        
        for p in percents:
            h = int(32 * (p / 100.0))
            variant = create_scale_variant(t_tight, h)
            name = f"train_{i}_p{p:03d}"
            res = eval_variant(name, variant, t_gt, model, transform, tokenizer, device, out_dir)
            res["eff_h"] = h
            res["percent"] = p
            train_results.append(res)
            
    print("\n=== MARKDOWN TABLE: sample_test.png ===\n")
    print("| Effective Text Height | Variant Size | Final Model Input | Prediction | Correct | CER | Notes |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in sample_results:
        eff_h = r.get("eff_h", "Control")
        notes = "Original" if "original" in r["name"] else "AutoCrop" if "autocrop" in r["name"] else ""
        print(f"| {eff_h} | {r['size']} | {r['tensor']} | {r['pred']} | {r['correct']} | {r['cer']:.4f} | {notes} |")

    print("\n=== MARKDOWN TABLE: Training Images ===\n")
    print("| Image | % Scale | Eff Height | Prediction | Correct | CER |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in train_results:
        pct = r.get("percent", "Control")
        eff_h = r.get("eff_h", "Control")
        img_name = r["name"].split('_p')[0] if "_p" in r["name"] else r["name"]
        print(f"| {img_name} | {pct}% | {eff_h} | {r['pred']} | {r['correct']} | {r['cer']:.4f} |")

if __name__ == "__main__":
    main()
