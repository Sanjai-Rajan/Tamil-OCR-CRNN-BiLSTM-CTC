import os
import csv
import json
from pathlib import Path
import random

def main():
    root = Path(__file__).resolve().parent.parent
    tamilchar_path = root / "data" / "TamilNet_old" / "TamilChar.csv"
    vocab_path = root / "data" / "tamil_ocr_dataset" / "vocabulary" / "tamil_vocab.json"
    out_dir = root / "data" / "tamil_char_clean"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not vocab_path.exists():
        print(f"Vocab missing: {vocab_path}")
        return
        
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        
    valid_chars = set(vocab.get("characters", []))
    
    with open(tamilchar_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()[1:] # Skip header
        
    valid_samples = []
    rejected = 0
    
    for line in lines:
        if not line: continue
        parts = line.split(',')
        label = parts[0].strip()
        img_name = parts[1].strip()
        
        # Determine if label can be composed strictly of valid atomic chars
        # For simplicity, we just check if every char in the label exists in vocab
        is_valid = True
        for c in label:
            if c not in valid_chars and c != " ":
                is_valid = False
                break
                
        if is_valid:
            valid_samples.append({"image": img_name, "label": label})
        else:
            rejected += 1
            
    # Deterministic shuffle & split
    valid_samples.sort(key=lambda x: x["image"])
    random.seed(42)
    random.shuffle(valid_samples)
    
    split_idx = int(len(valid_samples) * 0.8)
    train_split = valid_samples[:split_idx]
    val_split = valid_samples[split_idx:]
    
    with open(out_dir / "train.json", 'w', encoding='utf-8') as f:
        json.dump({"samples": train_split}, f, indent=4)
    with open(out_dir / "val.json", 'w', encoding='utf-8') as f:
        json.dump({"samples": val_split}, f, indent=4)
        
    print(f"TamilChar Dataset Prep Complete.")
    print(f"Total valid samples: {len(valid_samples)}")
    print(f"Train: {len(train_split)} | Val: {len(val_split)}")
    print(f"Rejected samples (OOV characters): {rejected}")

if __name__ == "__main__":
    main()
