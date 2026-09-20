import os
import sys
import argparse
import shutil
import json
import csv
from pathlib import Path

def setup_argparse():
    parser = argparse.ArgumentParser(description="Import TamilNet_old dataset into standardized packets.")
    parser.add_argument("--source", type=str, default="data/TamilNet_old", help="Path to TamilNet_old directory")
    parser.add_argument("--dest", type=str, default="data/tamil_ocr_dataset/imported/tamil", help="Destination imported root")
    parser.add_argument("--packet-size", type=int, default=5000, help="Number of images per packet")
    parser.add_argument("--dry-run", action="store_true", help="Do not copy files, just simulate")
    return parser

def load_tamilchar_mapping(csv_path):
    mapping = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header
        for row in reader:
            if len(row) >= 3:
                class_id = int(row[0].strip())
                unicode_hexes = row[2].strip().split()
                try:
                    char = "".join(chr(int(hex_str, 16)) for hex_str in unicode_hexes)
                    mapping[class_id] = char
                except ValueError:
                    print(f"Warning: Could not parse unicode {row[2]} for class {class_id}")
    return mapping

def process_split(source_root, dest_root, split, mapping, packet_size, dry_run):
    source_dir = source_root / split
    if not source_dir.exists():
        print(f"Warning: Source split directory {source_dir} not found.")
        return
        
    dest_split_dir = dest_root / split
    dest_split_dir.mkdir(parents=True, exist_ok=True)
    
    # Gather all images and their labels
    items = []
    
    if split == "train":
        # Iterate over usr_xx folders
        for usr_dir in source_dir.iterdir():
            if usr_dir.is_dir() and usr_dir.name.startswith("usr_"):
                for img_path in usr_dir.glob("*.tiff"):
                    # Filename format: {class_id}t{num}.tiff e.g. 000t01.tiff
                    filename = img_path.name
                    class_id_str = filename.split('t')[0]
                    try:
                        class_id = int(class_id_str)
                        if class_id in mapping:
                            items.append((img_path, mapping[class_id]))
                        else:
                            print(f"Warning: Class ID {class_id} not found in mapping for {img_path}")
                    except ValueError:
                        print(f"Warning: Could not parse class ID from {filename}")
    elif split == "test":
        # Load ground_truth.txt
        gt_path = source_root / "ground_truth.txt"
        if not gt_path.exists():
            print(f"Warning: ground_truth.txt not found at {gt_path}")
            return
            
        gt_map = {}
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    file_prefix = parts[0]
                    try:
                        class_id = int(parts[1])
                        gt_map[f"{file_prefix}.tiff"] = class_id
                    except ValueError:
                        pass
                        
        for img_path in source_dir.glob("*.tiff"):
            filename = img_path.name
            if filename in gt_map:
                class_id = gt_map[filename]
                if class_id in mapping:
                    items.append((img_path, mapping[class_id]))
                else:
                    print(f"Warning: Class ID {class_id} not found in mapping for {img_path}")
            else:
                print(f"Warning: {filename} not found in ground truth")

    total_items = len(items)
    print(f"Found {total_items} items in {split} split.")
    
    if total_items == 0:
        return
        
    # Sort for deterministic processing
    items.sort(key=lambda x: str(x[0]))
    
    # Chunk into packets
    num_packets = (total_items + packet_size - 1) // packet_size
    print(f"Will generate {num_packets} packets (max {packet_size} items each).")
    
    packet_idx = 1
    global_idx = 0
    
    for i in range(0, total_items, packet_size):
        chunk = items[i:i+packet_size]
        packet_name = f"packet_tamilnet_{packet_idx:03d}"
        packet_dir = dest_split_dir / packet_name
        images_dir = packet_dir / "images"
        
        if not dry_run:
            images_dir.mkdir(parents=True, exist_ok=True)
            manifest_fp = open(packet_dir / "manifest.jsonl", 'w', encoding='utf-8')
            
        for img_path, label in chunk:
            ext = img_path.suffix
            new_filename = f"tamilnet_{split}_{global_idx+1:06d}{ext}"
            new_img_path = images_dir / new_filename
            
            if not dry_run:
                shutil.copy2(img_path, new_img_path)
                manifest_entry = {
                    "original_image": str(img_path),
                    "image": f"images/{new_filename}",
                    "label": label,
                    "split": split,
                    "packet": packet_name
                }
                manifest_fp.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
                
            global_idx += 1
            
        if not dry_run:
            manifest_fp.close()
            
        print(f"Processed {packet_name}: {len(chunk)} images.")
        packet_idx += 1

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    
    source_root = Path(args.source)
    dest_root = Path(args.dest)
    
    if not source_root.exists():
        print(f"Error: Source directory {source_root} not found.")
        sys.exit(1)
        
    mapping_path = source_root / "TamilChar.csv"
    if not mapping_path.exists():
        print(f"Error: Mapping file {mapping_path} not found.")
        sys.exit(1)
        
    print("Loading class ID to character mapping...")
    mapping = load_tamilchar_mapping(mapping_path)
    print(f"Loaded {len(mapping)} class mappings.")
    
    if args.dry_run:
        print("\n=== DRY RUN MODE: No files will be created or copied ===\n")
        
    process_split(source_root, dest_root, "train", mapping, args.packet_size, args.dry_run)
    process_split(source_root, dest_root, "test", mapping, args.packet_size, args.dry_run)
    
    print("\nImport process completed.")
    if not args.dry_run:
        print("Please run `python scripts/build_tamil_vocabulary.py` to integrate the new characters.")

if __name__ == "__main__":
    main()
