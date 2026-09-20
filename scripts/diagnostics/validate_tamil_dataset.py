import os
import argparse
from pathlib import Path
import json
import csv
from PIL import Image

def setup_argparse():
    parser = argparse.ArgumentParser(description="Validate Tamil OCR Dataset")
    parser.add_argument("--dataset", type=str, default="data/tamil_ocr_dataset", help="Path to copied dataset")
    parser.add_argument("--language", type=str, default="tamil", help="Language prefix for files")
    return parser

def validate_split(dataset_dir, split, report_lines):
    split_dir = dataset_dir / "packets" / split
    if not split_dir.exists():
        report_lines.append(f"[{split.upper()}] Directory not found: {split_dir}")
        return False
        
    images_count = 0
    annotations_count = 0
    missing_images = 0
    missing_annotations = 0
    duplicate_filenames = 0
    empty_labels = 0
    invalid_images = 0
    
    seen_images = set()
    
    # Iterate through packets
    for packet_dir in sorted(split_dir.iterdir()):
        if not packet_dir.is_dir() or not packet_dir.name.startswith("packet_"):
            continue
            
        images_dir = packet_dir / "images"
        anno_file = packet_dir / "annotations.txt"
        
        if not images_dir.exists() or not anno_file.exists():
            continue
            
        # Check images
        packet_images = list(images_dir.glob("*.jpg"))
        images_count += len(packet_images)
        
        for img_path in packet_images:
            if img_path.name in seen_images:
                duplicate_filenames += 1
            seen_images.add(img_path.name)
            
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception:
                invalid_images += 1
                
        # Check annotations
        with open(anno_file, 'r', encoding='utf-8') as f:
            for line in f:
                if ',' not in line:
                    continue
                annotations_count += 1
                rel_img_path, label = line.strip().split(',', 1)
                
                # Check if image exists
                full_img_path = packet_dir / rel_img_path
                if not full_img_path.exists():
                    missing_images += 1
                    
                # Check label
                if not label.strip():
                    empty_labels += 1
                    
    # Check if every image has an annotation (simplification: counts match)
    if images_count != annotations_count:
        missing_annotations = abs(images_count - annotations_count)

    status = "PASS"
    if missing_images > 0 or missing_annotations > 0 or duplicate_filenames > 0 or empty_labels > 0 or invalid_images > 0:
        status = "FAIL"
        
    report_lines.append(f"{split.upper()}")
    report_lines.append(f"Images: {images_count}")
    report_lines.append(f"Annotations: {annotations_count}")
    report_lines.append(f"Missing images: {missing_images}")
    report_lines.append(f"Missing annotations: {missing_annotations}")
    report_lines.append(f"Duplicate filenames: {duplicate_filenames}")
    report_lines.append(f"Empty labels: {empty_labels}")
    report_lines.append(f"Invalid images: {invalid_images}")
    report_lines.append(f"Status: {status}\n")
    
    return status == "PASS"

def validate_dataset(args):
    dataset_dir = Path(args.dataset)
    report_lines = []
    
    print(f"--- Validating {args.language.upper()} OCR Dataset ---")
    
    splits = ['train', 'validation', 'test']
    all_passed = True
    
    for split in splits:
        if not validate_split(dataset_dir, split, report_lines):
            all_passed = False
            
    report_dir = dataset_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "dataset_validation_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    print(f"Validation complete. Report saved to {report_path}")
    print(f"Overall Status: {'PASS' if all_passed else 'FAIL'}")

if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    validate_dataset(args)
