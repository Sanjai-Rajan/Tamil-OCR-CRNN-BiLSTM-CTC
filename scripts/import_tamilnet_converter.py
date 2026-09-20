import os
import sys
import argparse
import shutil
import json
import csv
from pathlib import Path

# Add project root to sys.path to import tokenizer
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from models.digitalization.tokenizer import Tokenizer

def load_tamilchar_mapping(csv_path):
    mapping = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 3:
                class_id = int(row[0].strip())
                unicode_hexes = row[2].strip().split()
                try:
                    char = "".join(chr(int(hx, 16)) for hx in unicode_hexes)
                    mapping[class_id] = char
                except ValueError:
                    pass
    return mapping

def validate_tokenizer(mapping, tokenizer):
    failed = 0
    encodable = 0
    unique_labels = set(mapping.values())
    for label in unique_labels:
        try:
            tokenizer.encode(label)
            encodable += 1
        except Exception as e:
            failed += 1
            print(f"Tokenizer failed to encode label '{label}': {e}")
    return len(mapping), len(unique_labels), encodable, failed

def process_split(source_root, dest_root, split, mapping):
    source_dir = source_root / split
    packet_dir = dest_root / split / "packet_tamilnet"
    
    if packet_dir.exists():
        print(f"CONFLICT: Destination {packet_dir} already exists. Stopping.")
        sys.exit(1)
        
    images_dir = packet_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_fp = open(packet_dir / "manifest.jsonl", 'w', encoding='utf-8')
    
    source_count = 0
    converted_count = 0
    manifest_count = 0
    missing_images = 0
    invalid_labels = 0
    invalid_ids = 0
    ground_truth_count = 0
    
    # Track destination filenames to prevent collisions
    dest_filenames = set()
    
    if split == "train":
        for usr_dir in source_dir.iterdir():
            if usr_dir.is_dir() and usr_dir.name.startswith("usr_"):
                writer_id = usr_dir.name
                for img_path in usr_dir.glob("*.tiff"):
                    source_count += 1
                    filename = img_path.name
                    class_id_str = filename.split('t')[0]
                    try:
                        class_id = int(class_id_str)
                    except ValueError:
                        invalid_ids += 1
                        continue
                        
                    if class_id not in mapping:
                        invalid_labels += 1
                        continue
                        
                    label = mapping[class_id]
                    
                    new_filename = f"tamilnet_train_{usr_dir.name}_{filename}"
                    if new_filename in dest_filenames:
                        print(f"COLLISION: {new_filename} already exists!")
                        sys.exit(1)
                    dest_filenames.add(new_filename)
                    
                    new_img_path = images_dir / new_filename
                    shutil.copy2(img_path, new_img_path)
                    converted_count += 1
                    
                    entry = {
                        "original_image": str(img_path.as_posix()),
                        "image": f"images/{new_filename}",
                        "label": label,
                        "split": split,
                        "packet": "packet_tamilnet",
                        "source_dataset": "TamilNet",
                        "writer_id": writer_id,
                        "original_filename": filename,
                        "tamilnet_class_id": class_id
                    }
                    manifest_fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    manifest_count += 1
                    
    elif split == "test":
        gt_path = source_root / "ground_truth.txt"
        gt_map = {}
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    ground_truth_count += 1
                    try:
                        gt_map[f"{parts[0]}.tiff"] = int(parts[1])
                    except ValueError:
                        pass
                        
        for img_path in source_dir.glob("*.tiff"):
            source_count += 1
            filename = img_path.name
            
            if filename not in gt_map:
                missing_images += 1
                continue
                
            class_id = gt_map[filename]
            if class_id not in mapping:
                invalid_labels += 1
                continue
                
            label = mapping[class_id]
            
            new_filename = f"tamilnet_test_{filename}"
            if new_filename in dest_filenames:
                print(f"COLLISION: {new_filename} already exists!")
                sys.exit(1)
            dest_filenames.add(new_filename)
            
            new_img_path = images_dir / new_filename
            shutil.copy2(img_path, new_img_path)
            converted_count += 1
            
            entry = {
                "original_image": str(img_path.as_posix()),
                "image": f"images/{new_filename}",
                "label": label,
                "split": split,
                "packet": "packet_tamilnet",
                "source_dataset": "TamilNet",
                "original_filename": filename,
                "tamilnet_class_id": class_id
            }
            manifest_fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
            manifest_count += 1

    manifest_fp.close()
    
    stats = {
        "source_count": source_count,
        "converted_count": converted_count,
        "manifest_count": manifest_count,
        "missing_images": missing_images,
        "ground_truth_count": ground_truth_count,
        "invalid_labels": invalid_labels,
        "invalid_ids": invalid_ids
    }
    return stats

def main():
    source_root = Path("data/TamilNet_old")
    dest_root = Path("data/tamil_ocr_dataset/imported/tamil")
    vocab_path = "data/tamil_ocr_dataset/vocabulary/tamil_vocab.json"
    
    mapping_path = source_root / "TamilChar.csv"
    mapping = load_tamilchar_mapping(mapping_path)
    
    tokenizer = Tokenizer(vocab_path=vocab_path)
    num_classes, unique_labels, encodable, failed = validate_tokenizer(mapping, tokenizer)
    
    print("--- VALIDATION METRICS ---")
    print(f"TOTAL TAMILNET CLASSES: {num_classes}")
    print(f"TOTAL UNIQUE TAMILNET LABELS: {unique_labels}")
    print(f"ENCODABLE: {encodable}")
    print(f"FAILED: {failed}")
    print("--------------------------")
    
    if failed > 0:
        print(f"STOPPING: Tokenizer validation failed for {failed} labels.")
        sys.exit(1)
        
    print("Proceeding with conversion...")
    
    train_stats = process_split(source_root, dest_root, "train", mapping)
    print(f"Train stats: {train_stats}")
    
    test_stats = process_split(source_root, dest_root, "test", mapping)
    print(f"Test stats: {test_stats}")

if __name__ == "__main__":
    main()
