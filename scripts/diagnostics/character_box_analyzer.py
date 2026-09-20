import os
import json
import cv2
import numpy as np
import random
from collections import defaultdict

# 1. Dataset Source
base_dir = r"data\tamil_ocr_dataset\imported\tamil\test\packet_001"
manifest_path = os.path.join(base_dir, "manifest.jsonl")

# Deterministic sampling
random.seed(42)
num_samples = 200

samples = []
with open(manifest_path, "r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))
        
samples = random.sample(samples, min(num_samples, len(samples)))

# Output directories
out_dir = r"outputs\diagnostics\character_box_analysis"
vis_dir = os.path.join(out_dir, "visualizations")
data_dir = os.path.join(out_dir, "component_data")
sum_dir = os.path.join(out_dir, "summaries")
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(sum_dir, exist_ok=True)

# 2. Tokenizer (to count ground truth tokens)
import sys
sys.path.insert(0, '.')
try:
    from models.digitalization.tokenizer import Tokenizer
    tokenizer = Tokenizer(r"data\tamil_ocr_dataset\vocabulary\tamil_vocab.json")
except Exception as e:
    print("Tokenizer failed to load:", e)
    tokenizer = None

# Stats
all_stats = []
word_length_groups = {
    "short (1-5)": [],
    "medium (6-10)": [],
    "long (11-15)": [],
    "very long (16+)": []
}

for i, sample in enumerate(samples):
    img_path = os.path.join(base_dir, sample["image"])
    label = sample["label"]
    
    if not os.path.exists(img_path):
        continue
        
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filter noise
    raw_components = []
    img_h, img_w = img.shape[:2]
    
    for j in range(1, num_labels):
        x, y, w, h, area = stats[j]
        # Ignore microscopic noise (area < 5 pixels)
        if area > 5:
            raw_components.append({
                "id": j, "x": x, "y": y, "w": w, "h": h, "area": int(area),
                "cx": float(centroids[j][0]), "cy": float(centroids[j][1]),
                "aspect_ratio": float(w)/h
            })
            
    # Sort by x coordinate
    raw_components.sort(key=lambda c: c["x"])
    
    # Grouping (Conservative deterministic grouping)
    # If a component is vertically above/below another, or extremely close horizontally
    candidates = []
    used = set()
    
    # A simple graph based grouping for overlapping/touching horizontally
    # Two components belong to the same candidate if their x-intervals overlap significantly
    # or they are very close horizontally.
    
    def are_grouped(c1, c2, max_dist=3):
        x1, w1 = c1["x"], c1["w"]
        x2, w2 = c2["x"], c2["w"]
        # Overlap in x
        if max(x1, x2) < min(x1+w1, x2+w2) + max_dist:
            return True
        return False
        
    for j, c1 in enumerate(raw_components):
        if j in used: continue
        
        group = [c1]
        used.add(j)
        
        # Grow group
        changed = True
        while changed:
            changed = False
            for k, c2 in enumerate(raw_components):
                if k in used: continue
                # Check if c2 should be in this group
                if any(are_grouped(g, c2) for g in group):
                    group.append(c2)
                    used.add(k)
                    changed = True
                    
        # Calculate bounding box of group
        gx = min(g["x"] for g in group)
        gy = min(g["y"] for g in group)
        gxw = max(g["x"] + g["w"] for g in group)
        gyh = max(g["y"] + g["h"] for g in group)
        candidates.append({
            "components": len(group),
            "x": gx, "y": gy, "w": gxw-gx, "h": gyh-gy,
            "area": sum(g["area"] for g in group)
        })
        
    candidates.sort(key=lambda c: c["x"])
    
    # Visualization
    vis = img.copy()
    # Draw raw components in blue (dashed/thin)
    for c in raw_components:
        cv2.rectangle(vis, (c["x"], c["y"]), (c["x"]+c["w"], c["y"]+c["h"]), (255, 0, 0), 1)
        
    # Draw grouped candidates in green (thick)
    for k, c in enumerate(candidates):
        cv2.rectangle(vis, (c["x"], c["y"]), (c["x"]+c["w"], c["y"]+c["h"]), (0, 255, 0), 2)
        cv2.putText(vis, str(k+1), (c["x"], c["y"]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        
    cv2.imwrite(os.path.join(vis_dir, f"sample_{i:03d}.png"), vis)
    
    # ground truth tokens
    gt_tokens = len(tokenizer.encode(label)) if tokenizer else 0
    
    # length group
    if gt_tokens <= 5: grp = "short (1-5)"
    elif gt_tokens <= 10: grp = "medium (6-10)"
    elif gt_tokens <= 15: grp = "long (11-15)"
    else: grp = "very long (16+)"
    
    stat_item = {
        "index": i,
        "image": sample["image"],
        "label": label,
        "gt_tokens": gt_tokens,
        "raw_components": len(raw_components),
        "candidate_regions": len(candidates),
        "length_group": grp
    }
    all_stats.append(stat_item)
    word_length_groups[grp].append(stat_item)

# Save JSON data
with open(os.path.join(data_dir, "components.json"), "w") as f:
    json.dump(all_stats, f, indent=2)

# Summaries
total = len(all_stats)
avg_raw = sum(s["raw_components"] for s in all_stats) / total if total else 0
avg_cand = sum(s["candidate_regions"] for s in all_stats) / total if total else 0
med_raw = sorted([s["raw_components"] for s in all_stats])[total//2] if total else 0
single_comp = sum(1 for s in all_stats if s["candidate_regions"] == 1)
frag_comp = sum(1 for s in all_stats if s["raw_components"] > s["gt_tokens"] * 2) # Arbitrary fragmentation threshold

summary = {
    "total_samples": total,
    "avg_raw_components": avg_raw,
    "median_raw_components": med_raw,
    "avg_candidate_regions": avg_cand,
    "single_component_words": single_comp,
    "fragmented_component_words": frag_comp,
    "length_groups": {k: len(v) for k, v in word_length_groups.items()}
}

with open(os.path.join(sum_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Character Box Analyzer complete.")
print(json.dumps(summary, indent=2))
