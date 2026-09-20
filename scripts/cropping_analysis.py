import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import torch
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode
from models.digitalization.tokenizer import Tokenizer
from scripts.dataset_loader import AspectRatioPreservingResize
from torchvision import transforms

def find_content_bbox(img_array, threshold=240):
    # Foreground pixels are below the threshold (darker)
    fg_mask = img_array < threshold
    if not np.any(fg_mask):
        return 0, 0, img_array.shape[1]-1, img_array.shape[0]-1
        
    y_indices, x_indices = np.where(fg_mask)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    return x_min, y_min, x_max, y_max

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

def main():
    print("==================================================")
    print("STEP 1 - ANALYZE THE IMAGE")
    print("==================================================")
    
    img_path = "sample_test.png"
    orig_img = Image.open(img_path).convert('L')
    orig_arr = np.array(orig_img)
    
    h, w = orig_arr.shape
    total_pixels = h * w
    bg_pixels = np.sum(orig_arr >= 240)
    fg_pixels = total_pixels - bg_pixels
    orig_blank_pct = (bg_pixels / total_pixels) * 100
    
    x_min, y_min, x_max, y_max = find_content_bbox(orig_arr)
    
    print(f"Original Width: {w}")
    print(f"Original Height: {h}")
    print(f"Total Pixels: {total_pixels}")
    print(f"Background Pixels: {bg_pixels}")
    print(f"Foreground Pixels: {fg_pixels}")
    print(f"Blank-space %: {orig_blank_pct:.2f}%")
    print(f"Detected Bounding Box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
    
    # Draw bounding box on original image for visualization
    viz_img = orig_img.copy().convert('RGB')
    draw = ImageDraw.Draw(viz_img)
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=1)
    
    out_dir = Path("outputs/training/tamil/refinement/cropping_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_img.save(out_dir / "original_with_bbox.png")
    
    # Initialize model
    print("\nLoading Model...")
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
    
    orig_pred = run_inference(model, orig_img, transform, tokenizer, device)
    
    margins = [0, 2, 4, 8]
    results = []
    
    print("\n==================================================")
    print("STEP 2, 3, 4, 5 - TESTING MARGINS")
    print("==================================================")
    
    for m in margins:
        # Calculate new crop box with margins
        c_xmin = max(0, x_min - m)
        c_ymin = max(0, y_min - m)
        c_xmax = min(w - 1, x_max + m)
        c_ymax = min(h - 1, y_max + m)
        
        cropped_img = orig_img.crop((c_xmin, c_ymin, c_xmax + 1, c_ymax + 1))
        cropped_arr = np.array(cropped_img)
        
        c_h, c_w = cropped_arr.shape
        c_total = c_h * c_w
        c_bg = np.sum(cropped_arr >= 240)
        c_fg = c_total - c_bg
        c_blank_pct = (c_bg / c_total) * 100
        crop_ratio = c_total / total_pixels
        area_removed = total_pixels - c_total
        
        pred = run_inference(model, cropped_img, transform, tokenizer, device)
        
        # Save visualization
        cropped_img.save(out_dir / f"crop_margin_{m}px.png")
        
        results.append({
            'variant': f"Crop {m}px",
            'dimensions': f"{c_w}x{c_h}",
            'blank_pct': f"{c_blank_pct:.2f}%",
            'prediction': pred,
            'foreground': c_fg,
            'background': c_bg,
            'crop_ratio': f"{crop_ratio*100:.2f}%",
            'area_removed': area_removed
        })
        
    print("\n==================================================")
    print("STEP 8 - FINAL REPORT")
    print("==================================================")
    
    print(f"{'Variant':<15} | {'Dimensions':<15} | {'Blank %':<10} | {'Prediction'}")
    print("-" * 60)
    print(f"{'Original':<15} | {f'{w}x{h}':<15} | {orig_blank_pct:>8.2f}% | {orig_pred}")
    for res in results:
        print(f"{res['variant']:<15} | {res['dimensions']:<15} | {res['blank_pct']:>9} | {res['prediction']}")
        
if __name__ == '__main__':
    main()
