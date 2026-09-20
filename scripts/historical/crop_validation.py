import os
import random
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import glob

def find_content_bbox(img_array, threshold=240):
    fg_mask = img_array < threshold
    if not np.any(fg_mask):
        return 0, 0, img_array.shape[1]-1, img_array.shape[0]-1
        
    y_indices, x_indices = np.where(fg_mask)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    return x_min, y_min, x_max, y_max

def main():
    print("Finding 10 random images...")
    img_dir = "data/tamil_ocr_dataset/imported/tamil/train/packet_001/images/"
    all_images = glob.glob(img_dir + "*.jpg") + glob.glob(img_dir + "*.png")
    
    if len(all_images) < 10:
        print("Not enough images found!")
        return
        
    sample_images = random.sample(all_images, 10)
    out_dir = Path("outputs/training/tamil/refinement/crop_training_v1/previews/")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    margin = 2
    
    for i, img_path in enumerate(sample_images):
        orig_img = Image.open(img_path).convert('L')
        orig_arr = np.array(orig_img)
        h, w = orig_arr.shape
        
        x_min, y_min, x_max, y_max = find_content_bbox(orig_arr, threshold=240)
        
        c_xmin = max(0, x_min - margin)
        c_ymin = max(0, y_min - margin)
        c_xmax = min(w - 1, x_max + margin)
        c_ymax = min(h - 1, y_max + margin)
        
        # 1. Original with BBox
        viz_bbox = orig_img.copy().convert('RGB')
        draw = ImageDraw.Draw(viz_bbox)
        draw.rectangle([c_xmin, c_ymin, c_xmax, c_ymax], outline="red", width=1)
        
        # 2. Cropped
        cropped_img = orig_img.crop((c_xmin, c_ymin, c_xmax + 1, c_ymax + 1))
        
        # 3. Final 32px height
        c_w, c_h = cropped_img.size
        new_w = max(4, round(c_w * 32 / c_h))
        final_img = cropped_img.resize((new_w, 32), Image.Resampling.LANCZOS)
        
        # Save side-by-side (Original with Bbox, Cropped, Final 32px)
        total_width = w + c_w + new_w + 20
        max_height = max(h, c_h, 32)
        
        combined = Image.new('RGB', (total_width, max_height), color='white')
        combined.paste(viz_bbox, (0, 0))
        combined.paste(cropped_img.convert('RGB'), (w + 10, 0))
        combined.paste(final_img.convert('RGB'), (w + c_w + 20, 0))
        
        fname = f"validation_{i:02d}.png"
        combined.save(out_dir / fname)
        print(f"Saved {fname} (orig: {w}x{h}, crop: {c_w}x{c_h}, final: {new_w}x32)")

if __name__ == '__main__':
    main()
