import sys
import os
import json
from pathlib import Path
from collections import Counter
import csv
import matplotlib.pyplot as plt
import cv2
import numpy as np

sys.path.insert(0, '.')
from models.digitalization.tokenizer import Tokenizer

def get_bucket(target_len):
    if target_len <= 5: return "1-5"
    if target_len <= 10: return "6-10"
    if target_len <= 15: return "11-15"
    if target_len <= 19: return "16-19" # Helper for logic, but output requires 16+ which groups everything >= 16
    return "20+"

def get_bucket_display(target_len):
    if target_len <= 5: return "1-5"
    if target_len <= 10: return "6-10"
    if target_len <= 15: return "11-15"
    if target_len <= 19: return "16-19"
    if target_len <= 24: return "20-24"
    if target_len <= 29: return "25-29"
    return "30+"

def main():
    working_root = "data/tamil_ocr_dataset"
    vocab_path = os.path.join(working_root, "vocabulary", "tamil_vocab.json")
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    train_dir = Path(working_root) / "imported" / "tamil" / "train"
    out_dir = Path("outputs/diagnostics/training_length_distribution")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    print("Loading training set manifests...")
    for packet_dir in train_dir.iterdir():
        if packet_dir.is_dir() and packet_dir.name.startswith("packet_"):
            manifest_path = packet_dir / "manifest.jsonl"
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        label = data["label"]
                        # Tokenize to get target token length
                        tokens = tokenizer.encode(label)
                        target_length = len(tokens)
                        
                        img_path = packet_dir / data["image"]
                        
                        # We need width and height. For efficiency, only read if it exists or use some bounding box data if in manifest.
                        # Since it's diagnostic and fast enough, let's read the image shapes. But 126k images might take 10 minutes to cv2.imread.
                        # Wait! In the manifest, there might not be width/height.
                        # Let's peek at one manifest first!
                        samples.append({
                            "path": str(img_path),
                            "label": label,
                            "length": target_length,
                            "bucket": get_bucket_display(target_length)
                        })

    print(f"Loaded {len(samples)} samples. Computing image widths... (this may take a few minutes)")
    
    # Actually, opening 126k images will take ~2-5 minutes. Let's do it efficiently.
    for s in samples:
        # Just read the header/shape without decoding the whole JPEG if possible, but cv2.imread is fast enough on SSD
        img = cv2.imread(s["path"], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            s["height"], s["width"] = img.shape
        else:
            s["height"], s["width"] = 0, 0
            
    # Filter valid
    samples = [s for s in samples if s["width"] > 0]
    total_samples = len(samples)
    print(f"Total valid training samples: {total_samples}")
    
    if total_samples == 0:
        print("No samples found.")
        return

    # Calculate frequencies
    lengths = [s["length"] for s in samples]
    max_length = max(lengths) if lengths else 0
    freq = Counter(lengths)
    
    with open(out_dir / "training_length_frequency.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target_length", "sample_count", "percentage"])
        for length in range(1, max_length + 1):
            count = freq.get(length, 0)
            pct = count / total_samples * 100
            writer.writerow([length, count, pct])
            
    # Calculate bucket statistics
    # Define exact requested buckets
    bucket_ranges = {
        "1-5": lambda l: 1 <= l <= 5,
        "6-10": lambda l: 6 <= l <= 10,
        "11-15": lambda l: 11 <= l <= 15,
        "16+": lambda l: l >= 16,
        "20+": lambda l: l >= 20,
        "25+": lambda l: l >= 25,
        "30+": lambda l: l >= 30,
    }
    
    bucket_stats = {}
    for name, condition in bucket_ranges.items():
        bucket_samples = [s for s in samples if condition(s["length"])]
        count = len(bucket_samples)
        pct = count / total_samples * 100
        
        if count > 0:
            b_lens = [s["length"] for s in bucket_samples]
            b_widths = [s["width"] for s in bucket_samples]
            b_w_per_t = [s["width"] / s["length"] for s in bucket_samples]
            
            bucket_stats[name] = {
                "bucket": name,
                "sample_count": count,
                "percentage": pct,
                "avg_target_length": np.mean(b_lens),
                "median_target_length": np.median(b_lens),
                "min_target_length": min(b_lens),
                "max_target_length": max(b_lens),
                "avg_image_width": np.mean(b_widths),
                "median_image_width": np.median(b_widths),
                "avg_width_per_token": np.mean(b_w_per_t)
            }
        else:
            bucket_stats[name] = {
                "bucket": name,
                "sample_count": 0,
                "percentage": 0.0,
                "avg_target_length": 0, "median_target_length": 0, "min_target_length": 0, "max_target_length": 0,
                "avg_image_width": 0, "median_image_width": 0, "avg_width_per_token": 0
            }
            
    with open(out_dir / "training_length_buckets.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bucket_stats["1-5"].keys())
        writer.writerow(dict((k,k) for k in bucket_stats["1-5"].keys())) # header
        for b in ["1-5", "6-10", "11-15", "16+", "20+", "25+", "30+"]:
            writer.writerow(bucket_stats[b])
            
    # Visualizations
    plt.figure(figsize=(10,6))
    plt.bar(range(1, max_length + 1), [freq.get(l, 0) for l in range(1, max_length + 1)])
    plt.xlabel("Target Token Length")
    plt.ylabel("Sample Count")
    plt.title("Training Target Length Distribution")
    plt.savefig(out_dir / "training_target_length_distribution.png")
    plt.close()
    
    plt.figure(figsize=(8,6))
    b_names = ["1-5", "6-10", "11-15", "16+"]
    b_pcts = [bucket_stats[b]["percentage"] for b in b_names]
    plt.bar(b_names, b_pcts)
    plt.xlabel("Length Bucket")
    plt.ylabel("Percentage of Training Set")
    plt.title("Training Set Composition by Length")
    plt.savefig(out_dir / "training_length_buckets.png")
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.scatter([s["length"] for s in samples[::10]], [s["width"] for s in samples[::10]], alpha=0.1) # 10% sample for plot speed
    plt.xlabel("Target Token Length")
    plt.ylabel("Image Width (px)")
    plt.title("Image Width vs Target Token Length (10% Sample)")
    plt.savefig(out_dir / "training_width_vs_target_length.png")
    plt.close()
    
    # Cumulative Distribution
    sorted_freq = [freq.get(l, 0) for l in range(1, max_length + 1)]
    cum_freq = np.cumsum(sorted_freq) / total_samples * 100
    plt.figure(figsize=(10,6))
    plt.plot(range(1, max_length + 1), cum_freq, marker='.')
    plt.xlabel("Target Token Length")
    plt.ylabel("Cumulative Percentage (%)")
    plt.title("Cumulative Training Samples by Length")
    plt.grid(True)
    plt.savefig(out_dir / "training_samples_by_length.png")
    plt.close()

    # Representative examples
    rep_names = {
        "shortest": lambda l: l <= 2,
        "1-5": lambda l: 3 <= l <= 5,
        "6-10": lambda l: 6 <= l <= 10,
        "11-15": lambda l: 11 <= l <= 15,
        "16+": lambda l: 16 <= l <= 19,
        "longest": lambda l: l >= 20
    }
    
    with open(out_dir / "representative_samples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "path", "label", "length"])
        import random
        random.seed(42)
        for cat, cond in rep_names.items():
            pool = [s for s in samples if cond(s["length"])]
            if pool:
                selected = random.choice(pool)
                writer.writerow([cat, selected["path"], selected["label"], selected["length"]])

    print("Done!")

if __name__ == '__main__':
    main()
