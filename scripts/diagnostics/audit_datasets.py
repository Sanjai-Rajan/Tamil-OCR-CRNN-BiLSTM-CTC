import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import json
import csv
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
import hashlib

def get_vocab():
    vocab_path = "data/tamil_ocr_dataset/vocabulary/tamil_vocab.json"
    if not os.path.exists(vocab_path):
        return set()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return set(data.get("characters", []))

VALID_CHARS = get_vocab()

def check_unicode(text):
    invalid = set()
    for char in text:
        if char not in VALID_CHARS:
            invalid.add(char)
    return invalid

def scan_datasets():
    datasets = []
    roots = [Path("data"), Path("dataset")]
    
    for root in roots:
        if not root.exists(): continue
        for r, d, f in os.walk(root):
            # Look for manifest.jsonl
            if "manifest.jsonl" in f:
                datasets.append({
                    "name": Path(r).name,
                    "path": Path(r),
                    "type": "jsonl",
                    "manifest": Path(r) / "manifest.jsonl"
                })
            # Look for labels.json
            elif "labels.json" in f:
                datasets.append({
                    "name": Path(r).name,
                    "path": Path(r),
                    "type": "json",
                    "manifest": Path(r) / "labels.json"
                })
    return datasets

def main():
    out_dir = Path("outputs/pre_5090_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = scan_datasets()
    
    inventory = []
    quality_issues = defaultdict(list)
    geometry_stats = {}
    
    # Leakage tracking
    global_labels = defaultdict(list)
    global_image_names = defaultdict(list)
    
    for ds in datasets:
        print(f"Scanning {ds['path']}...")
        img_count = 0
        missing_images = 0
        empty_labels = 0
        invalid_unicode_samples = 0
        
        heights = []
        widths = []
        aspect_ratios = []
        
        # We will sample geometry to save time (max 500)
        max_geom_samples = 500
        geom_sampled = 0
        
        with open(ds['manifest'], 'r', encoding='utf-8') as f:
            if ds['type'] == 'jsonl':
                for line_idx, line in enumerate(f):
                    if not line.strip(): continue
                    try:
                        entry = json.loads(line)
                        img_path = ds['path'] / entry.get('image', '')
                        label = entry.get('label', '')
                        
                        img_count += 1
                        
                        # Leakage
                        global_labels[label].append(f"{ds['path']}::{entry.get('image')}")
                        global_image_names[entry.get('image', '')].append(str(ds['path']))
                        
                        # Quality
                        if not img_path.exists():
                            missing_images += 1
                        if not label.strip():
                            empty_labels += 1
                        else:
                            invalid = check_unicode(label)
                            if invalid:
                                invalid_unicode_samples += 1
                                if invalid_unicode_samples <= 5:
                                    quality_issues[str(ds['path'])].append(f"Invalid chars {invalid} in label '{label}'")
                        
                        # Geometry
                        if geom_sampled < max_geom_samples and img_path.exists():
                            try:
                                with Image.open(img_path) as img:
                                    w, h = img.size
                                    heights.append(h)
                                    widths.append(w)
                                    aspect_ratios.append(w/h if h > 0 else 0)
                                    geom_sampled += 1
                            except:
                                pass
                    except Exception as e:
                        quality_issues[str(ds['path'])].append(f"JSON parse error line {line_idx}")
            
        inventory.append({
            "Path": str(ds['path']),
            "Format": ds['type'],
            "Total Images": img_count,
            "Missing Images": missing_images,
            "Empty Labels": empty_labels,
            "OOV Samples": invalid_unicode_samples
        })
        
        if heights:
            geometry_stats[str(ds['path'])] = {
                "Mean Height": np.mean(heights),
                "Mean Width": np.mean(widths),
                "Mean Aspect Ratio": np.mean(aspect_ratios),
                "Min Height": np.min(heights),
                "Max Height": np.max(heights)
            }
            
    # Write INVENTORY CSV
    with open(out_dir / "DATASET_INVENTORY.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Path", "Format", "Total Images", "Missing Images", "Empty Labels", "OOV Samples"])
        writer.writeheader()
        writer.writerows(inventory)
        
    # Write INVENTORY MD
    with open(out_dir / "DATASET_INVENTORY.md", "w", encoding="utf-8") as f:
        f.write("# Pre-5090 Audit: Phase 5 - Dataset Inventory\n\n")
        f.write("| Path | Format | Total Images | Missing Images | Empty Labels | OOV Samples |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for inv in inventory:
            f.write(f"| {inv['Path']} | {inv['Format']} | {inv['Total Images']} | {inv['Missing Images']} | {inv['Empty Labels']} | {inv['OOV Samples']} |\n")
            
    # Write QUALITY REPORT
    with open(out_dir / "DATASET_QUALITY_REPORT.md", "w", encoding="utf-8") as f:
        f.write("# Pre-5090 Audit: Phase 6 - Data Quality Report\n\n")
        for inv in inventory:
            f.write(f"## {inv['Path']}\n")
            f.write(f"- Missing Images: {inv['Missing Images']}\n")
            f.write(f"- Empty Labels: {inv['Empty Labels']}\n")
            f.write(f"- Out-Of-Vocabulary Samples: {inv['OOV Samples']}\n")
            issues = quality_issues.get(inv['Path'], [])
            if issues:
                f.write("- Sample Issues:\n")
                for iss in issues:
                    f.write(f"  - {iss}\n")
            f.write("\n")
            
    # Write GEOMETRY REPORT
    with open(out_dir / "DATASET_GEOMETRY_REPORT.md", "w", encoding="utf-8") as f:
        f.write("# Pre-5090 Audit: Phase 7 - Geometry Report (Sampled)\n\n")
        f.write("| Path | Mean Height | Min H | Max H | Mean Width | Mean AR |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for path, stats in geometry_stats.items():
            f.write(f"| {path} | {stats['Mean Height']:.1f} | {stats['Min Height']} | {stats['Max Height']} | {stats['Mean Width']:.1f} | {stats['Mean Aspect Ratio']:.2f} |\n")
            
    # Write LEAKAGE AUDIT
    with open(out_dir / "LEAKAGE_AUDIT.md", "w", encoding="utf-8") as f:
        f.write("# Pre-5090 Audit: Phase 8 - Duplicate / Leakage Audit\n\n")
        
        f.write("## 1. Image Name Collisions (Cross-Dataset)\n")
        collisions = {k: v for k, v in global_image_names.items() if len(set(v)) > 1}
        f.write(f"Found {len(collisions)} images with the same filename across DIFFERENT dataset directories.\n")
        
        f.write("\n## 2. Label Duplication (Cross-Dataset Leakage Risk)\n")
        # Check if same exact label string exists in multiple distinct datasets
        cross_ds_labels = 0
        for label, paths in global_labels.items():
            unique_dirs = set(p.split("::")[0] for p in paths)
            if len(unique_dirs) > 1:
                cross_ds_labels += 1
                
        f.write(f"Found {cross_ds_labels} distinct label text strings that appear in MULTIPLE dataset directories (e.g., both Train and Test directories).\n")
        f.write("*(Note: Highly frequent short words may naturally overlap between Train/Test, but exact long strings shouldn't).* \n")

if __name__ == "__main__":
    main()
