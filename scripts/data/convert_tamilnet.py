import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import csv
import json
from pathlib import Path

def get_vocab():
    vocab_path = Path("data/tamil_ocr_dataset/vocabulary/tamil_vocab.json")
    if not vocab_path.exists():
        return set()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return set(data.get("characters", []))

def main():
    base_dir = Path("data/TamilNet_old")
    csv_path = base_dir / "TamilChar.csv"
    gt_path = base_dir / "ground_truth.txt"
    test_dir = base_dir / "test"
    
    out_dir = Path("outputs/pre_5090_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    imported_dir = Path("data/tamil_ocr_dataset/imported/tamil_old_converted")
    imported_dir.mkdir(parents=True, exist_ok=True)
    
    vocab = get_vocab()
    
    # 1. Parse TamilChar.csv
    class_map = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = row['Class Id'].strip()
            char = row['Tamil Character'].strip()
            class_map[class_id] = char
            
    # 2. Parse ground_truth.txt
    missing_images = 0
    missing_labels = 0
    malformed_lines = 0
    duplicate_entries = 0
    
    unique_unicode = set()
    oov_chars = set()
    
    seen_images = set()
    
    manifest_entries = []
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 2:
                malformed_lines += 1
                continue
                
            img_id, class_id = parts
            img_file = f"{img_id}.tiff"
            img_path = test_dir / img_file
            
            if img_file in seen_images:
                duplicate_entries += 1
            seen_images.add(img_file)
            
            if not img_path.exists():
                missing_images += 1
                
            if class_id not in class_map:
                missing_labels += 1
                label = ""
            else:
                label = class_map[class_id]
                
            for char in label:
                unique_unicode.add(char)
                if char not in vocab:
                    oov_chars.add(char)
                    
            entry = {
                "image": str(img_path.resolve()),
                "label": label
            }
            manifest_entries.append(entry)
            
    # Write manifest
    manifest_path = imported_dir / "manifest.jsonl"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    # Write report
    report_path = out_dir / "TAMILNET_OLD_INGESTION_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Task 2: TamilNet_old Ingestion Report\n\n")
        f.write("## 1. Source Structure\n")
        f.write("- **Dataset**: `data/TamilNet_old/`\n")
        f.write("- **Format**: `ground_truth.txt` maps image prefixes to Class IDs. `TamilChar.csv` maps Class IDs to Unicode strings.\n")
        f.write(f"- **Total lines parsed**: {len(manifest_entries)}\n\n")
        
        f.write("## 2. Validation Metrics\n")
        f.write(f"- **Missing Images**: {missing_images}\n")
        f.write(f"- **Missing Labels**: {missing_labels}\n")
        f.write(f"- **Malformed Lines**: {malformed_lines}\n")
        f.write(f"- **Duplicate Entries**: {duplicate_entries}\n\n")
        
        f.write("## 3. Unicode & Vocabulary Audit\n")
        f.write(f"- **Unique Unicode Characters Found**: {len(unique_unicode)}\n")
        f.write(f"- **Characters Outside Current Vocabulary**: {len(oov_chars)}\n")
        if oov_chars:
            f.write("  - OOV Characters: " + " ".join(oov_chars) + "\n")
        
        f.write("\n## 4. Output\n")
        f.write(f"- Successfully generated `{manifest_path}`\n")
        
if __name__ == "__main__":
    main()
