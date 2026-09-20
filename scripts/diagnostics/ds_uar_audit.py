import os
import json
import glob
import pandas as pd
from PIL import Image
import hashlib
from pathlib import Path
from collections import defaultdict
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = str(PROJECT_ROOT.parent.parent / "DS UAR")
OUT_DIR = str(PROJECT_ROOT / "outputs/dataset_source_audit")
VOCAB_PATH = str(PROJECT_ROOT / "data/tamil_ocr_dataset/vocabulary/tamil_vocab.json")

os.makedirs(OUT_DIR, exist_ok=True)

# Load existing vocab
if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        vocab_chars = set(vocab_data.get('characters', []))
else:
    vocab_chars = set()

def get_tree_skeleton(root_path, max_depth=3):
    tree_lines = []
    def walk(curr_path, prefix, depth):
        if depth > max_depth:
            tree_lines.append(f"{prefix}...")
            return
        try:
            items = sorted(os.listdir(curr_path))
        except PermissionError:
            return
        
        # Group image files to avoid clutter
        image_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        images = [i for i in items if os.path.splitext(i)[1].lower() in image_exts]
        others = [i for i in items if i not in images]
        
        if images:
            tree_lines.append(f"{prefix}├── <{len(images)} image files>")
            
        for i, item in enumerate(others):
            is_last = (i == len(others) - 1)
            connector = "└── " if is_last else "├── "
            item_path = os.path.join(curr_path, item)
            tree_lines.append(f"{prefix}{connector}{item}/" if os.path.isdir(item_path) else f"{prefix}{connector}{item}")
            if os.path.isdir(item_path):
                extension = "    " if is_last else "│   "
                walk(item_path, prefix + extension, depth + 1)
                
    tree_lines.append(f"{os.path.basename(root_path)}/")
    walk(root_path, "", 1)
    return "\n".join(tree_lines)

def analyze_dataset(ds_path):
    info = {
        "path": ds_path,
        "name": os.path.basename(ds_path),
        "total_files": 0,
        "image_files": 0,
        "extensions": defaultdict(int),
        "class_folders": set(),
        "annotation_files": [],
        "samples": [],
        "approx_size_bytes": 0
    }
    
    image_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    
    for root, dirs, files in os.walk(ds_path):
        for f in files:
            info["total_files"] += 1
            ext = os.path.splitext(f)[1].lower()
            info["extensions"][ext] += 1
            fpath = os.path.join(root, f)
            try:
                info["approx_size_bytes"] += os.path.getsize(fpath)
            except:
                pass
                
            if ext in image_exts:
                info["image_files"] += 1
                rel_dir = os.path.relpath(root, ds_path)
                if rel_dir != ".":
                    info["class_folders"].add(rel_dir)
                if random.random() < 0.05 and len(info["samples"]) < 50:
                    info["samples"].append(fpath)
            elif ext in {'.csv', '.xlsx', '.xls', '.txt', '.json', '.xml'}:
                info["annotation_files"].append(fpath)
                
    info["class_folders"] = list(info["class_folders"])
    info["extensions"] = dict(info["extensions"])
    
    # Analyze geometry
    widths = []
    heights = []
    channels = set()
    for img_path in info["samples"]:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                channels.add(img.mode)
        except:
            pass
            
    if widths:
        info["geometry"] = {
            "min_w": min(widths),
            "max_w": max(widths),
            "med_w": sorted(widths)[len(widths)//2],
            "min_h": min(heights),
            "max_h": max(heights),
            "med_h": sorted(heights)[len(heights)//2],
            "channels": list(channels)
        }
    else:
        info["geometry"] = {}
        
    return info

datasets = {}
for name in ["1", "2", "3", "4", "5", "6", "7", "8", "data"]:
    p = os.path.join(SOURCE_DIR, name)
    if os.path.exists(p):
        print(f"Analyzing {name}...")
        datasets[name] = analyze_dataset(p)

with open(os.path.join(OUT_DIR, "raw_audit.json"), "w", encoding="utf-8") as f:
    json.dump(datasets, f, indent=2)

tree = get_tree_skeleton(SOURCE_DIR, max_depth=3)
with open(os.path.join(OUT_DIR, "DS_UAR_COMPLETE_TREE.md"), "w", encoding="utf-8") as f:
    f.write("# DS UAR Complete Tree\n```text\n" + tree + "\n```")

print("Audit raw collection complete.")
