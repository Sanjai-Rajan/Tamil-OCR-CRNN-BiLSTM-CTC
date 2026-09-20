import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from pathlib import Path

def get_content_bbox(img_np, threshold=240):
    fg_mask = img_np < threshold
    y_indices, x_indices = np.where(fg_mask)
    if len(y_indices) == 0:
        return 0, 0, 0, 0, 0
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    content_w = x_max - x_min + 1
    content_h = y_max - y_min + 1
    fg_pct = np.sum(fg_mask) / (img_np.shape[0] * img_np.shape[1]) * 100
    return content_w, content_h, x_min, y_min, fg_pct

def analyze_dataset(manifest_path, img_dir, num_samples=500):
    widths = []
    heights = []
    aspect_ratios = []
    fg_pcts = []
    text_img_h_ratios = []
    tight_crops = []
    
    count = 0
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line.strip())
            path = os.path.join(img_dir, data["image"])
            if not os.path.exists(path): continue
            
            try:
                img = Image.open(path).convert('L')
                img_np = np.array(img)
                h, w = img_np.shape
                
                content_w, content_h, x_min, y_min, fg_pct = get_content_bbox(img_np)
                
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / max(1, h))
                fg_pcts.append(fg_pct)
                text_img_h_ratios.append(content_h / max(1, h))
                
                # Check if tightly cropped (margin <= 4 pixels total)
                tight = (w - content_w <= 4) and (h - content_h <= 4)
                tight_crops.append(tight)
                
                count += 1
                if count >= num_samples:
                    break
            except Exception as e:
                pass
                
    def print_stats(name, arr):
        arr = np.array(arr)
        print(f"--- {name} ---")
        print(f"Min: {np.min(arr):.2f}, Max: {np.max(arr):.2f}, Mean: {np.mean(arr):.2f}, Median: {np.median(arr):.2f}")
        print(f"P10: {np.percentile(arr, 10):.2f}, P25: {np.percentile(arr, 25):.2f}, P50: {np.percentile(arr, 50):.2f}, P75: {np.percentile(arr, 75):.2f}, P90: {np.percentile(arr, 90):.2f}, P95: {np.percentile(arr, 95):.2f}")
        
    print("==================================================")
    print(f"PHASE 1 - TRAINING DATA ANALYSIS ({count} samples)")
    print("==================================================")
    print_stats("Width", widths)
    print_stats("Height", heights)
    print_stats("Aspect Ratio", aspect_ratios)
    print_stats("Foreground Percentage", fg_pcts)
    print_stats("Text-to-Image Height Ratio", text_img_h_ratios)
    print(f"\nImages tightly cropped: {sum(tight_crops)} / {count} ({(sum(tight_crops)/count)*100:.2f}%)")
    
def analyze_real_world():
    print("\n==================================================")
    print("PHASE 2 & 3 - REAL-WORLD & SAMPLE TEST ANALYSIS")
    print("==================================================")
    
    test_images = ['sample_test.png', 'sample.jpg']
    for img_path in test_images:
        if not os.path.exists(img_path): continue
        img = Image.open(img_path).convert('L')
        img_np = np.array(img)
        h, w = img_np.shape
        
        content_w, content_h, x_min, y_min, fg_pct = get_content_bbox(img_np)
        
        print(f"\nImage: {img_path}")
        print(f"Dimensions: {w}x{h}")
        print(f"Content BBox: w={content_w}, h={content_h}, x={x_min}, y={y_min}")
        print(f"Text-to-Image Height Ratio: {content_h / max(1, h):.4f}")
        print(f"Foreground Percentage: {fg_pct:.2f}%")

if __name__ == "__main__":
    manifest = "data/tamil_ocr_dataset/imported/tamil/train/packet_001/manifest.jsonl"
    img_dir = "data/tamil_ocr_dataset/imported/tamil/train/packet_001"
    analyze_dataset(manifest, img_dir, num_samples=500)
    analyze_real_world()
