import sys
import os
import argparse
import shutil
import json
import yaml
import csv
from pathlib import Path
from PIL import Image

def setup_argparse():
    parser = argparse.ArgumentParser(description="Staged Import of Tamil OCR Dataset Packet")
    parser.add_argument("--language", type=str, default="tamil", help="Language prefix for files")
    parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"], help="Dataset split")
    parser.add_argument("--packet", type=int, required=True, help="Packet number (1-indexed)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without copying")
    parser.add_argument("--force", action="store_true", help="Force overwrite if packet already exists")
    return parser

def load_config(config_path="config/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_annotations(annotation_file):
    annotations = []
    if not annotation_file.exists():
        return annotations
    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            if ',' in line:
                img_path, label = line.strip().split(',', 1)
                annotations.append({"img_path": img_path.strip(), "label": label.strip()})
    return annotations

def validate_packet(packet_dir, expected_samples):
    images_dir = packet_dir / "images"
    anno_file = packet_dir / "annotations.txt"
    
    report = {
        "expected": expected_samples,
        "images_copied": 0,
        "annotations": 0,
        "missing_images": 0,
        "missing_labels": 0,
        "invalid_images": 0,
        "duplicate_names": 0,
        "status": "PASS"
    }
    
    if not images_dir.exists() or not anno_file.exists():
        report["status"] = "FAIL"
        return report
        
    packet_images = list(images_dir.glob("*.jpg"))
    report["images_copied"] = len(packet_images)
    
    seen = set()
    for img in packet_images:
        if img.name in seen:
            report["duplicate_names"] += 1
        seen.add(img.name)
        try:
            with Image.open(img) as im:
                im.verify()
        except Exception:
            report["invalid_images"] += 1
            
    with open(anno_file, 'r', encoding='utf-8') as f:
        for line in f:
            if ',' in line:
                report["annotations"] += 1
                rel_path, label = line.strip().split(',', 1)
                if not (packet_dir / rel_path).exists():
                    report["missing_images"] += 1
                if not label.strip():
                    report["missing_labels"] += 1
                    
    if (report["images_copied"] != report["expected"] or 
        report["annotations"] != report["expected"] or
        report["missing_images"] > 0 or 
        report["missing_labels"] > 0 or 
        report["invalid_images"] > 0 or 
        report["duplicate_names"] > 0):
        report["status"] = "FAIL"
        
    return report

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    config = load_config()
    
    source_root = Path(config["dataset"]["source_root"])
    working_root = Path(config["dataset"]["working_root"])
    packet_size = config["dataset"]["packet_size"]
    
    if not source_root.exists():
        print(f"Error: Source root {source_root} does not exist.")
        return
        
    # Map split names to source split names (val in source -> validation in dest)
    source_split_map = {"train": "train", "validation": "val", "test": "test"}
    source_split = source_split_map[args.split]
    
    anno_file = source_root / "txt_files" / source_split / f"{source_split}_augmented.txt"
    if not anno_file.exists():
        print(f"Error: Annotation file {anno_file} not found.")
        return
        
    annotations = load_annotations(anno_file)
    total_records = len(annotations)
    
    start_idx = (args.packet - 1) * packet_size
    end_idx = min(start_idx + packet_size, total_records)
    
    if start_idx >= total_records:
        print(f"Error: Packet {args.packet} is out of bounds (only {total_records} total records).")
        return
        
    packet_annotations = annotations[start_idx:end_idx]
    actual_samples = len(packet_annotations)
    
    packet_name = f"packet_{args.packet:03d}"
    imported_dir = working_root / "imported" / args.language / args.split / packet_name
    
    # Check registry
    registry_dir = working_root / "manifests" / args.language
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / "packet_registry.json"
    
    registry = {"language": args.language, "train": {}, "validation": {}, "test": {}}
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
            
    # Check if already imported
    is_imported = registry[args.split].get(packet_name, {}).get("status") == "imported"
    if is_imported and not args.dry_run and not args.force:
        print(f"Packet {args.packet:03d} already imported.")
        print("Skipping duplicate import. Use --force to overwrite.")
        return
    
    if args.dry_run:
        print(f"\n--- DRY RUN: Importing {args.language.upper()} OCR Dataset ---")
        print(f"Source: {source_root}")
        print(f"Destination: {imported_dir}")
        print(f"Packet: {args.packet:03d}")
        print(f"Images to copy: {actual_samples}")
        print(f"Annotation records: {actual_samples}")
        print("No files will be modified.")
        return
        
    print(f"\n--- IMPORTING {args.language.upper()} OCR Dataset ---")
    print(f"Packet: {args.packet:03d}")
    print(f"Destination: {imported_dir}")
    
    if imported_dir.exists():
        shutil.rmtree(imported_dir)
        
    images_dir = imported_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    packet_anno_fp = open(imported_dir / "annotations.txt", 'w', encoding='utf-8')
    packet_manifest_fp = open(imported_dir / "manifest.jsonl", 'w', encoding='utf-8')
    
    # Also update global mapping incrementally (optional but good for tracking)
    mappings_dir = working_root / "mappings" / args.language
    mappings_dir.mkdir(parents=True, exist_ok=True)
    mapping_csv_path = mappings_dir / f"{args.split}_mapping.csv"
    mode = 'a' if mapping_csv_path.exists() else 'w'
    mapping_csv = open(mapping_csv_path, mode, newline='', encoding='utf-8')
    mapping_writer = csv.writer(mapping_csv)
    if mode == 'w':
        mapping_writer.writerow(["original_filename", "new_filename", "label", "split", "packet"])
    
    for i, anno in enumerate(packet_annotations):
        global_idx = start_idx + i
        original_img_rel_path = anno["img_path"] 
        original_img_path = source_root / original_img_rel_path
        
        if not original_img_path.exists():
            print(f"Warning: Missing source image {original_img_path}")
            continue
            
        ext = original_img_path.suffix
        new_filename = f"{args.language}_{args.split}_{global_idx+1:06d}{ext}"
        new_img_path = images_dir / new_filename
        
        shutil.copy2(original_img_path, new_img_path)
        
        packet_anno_fp.write(f"images/{new_filename},{anno['label']}\n")
        
        manifest_entry = {
            "original_image": str(original_img_path),
            "image": f"images/{new_filename}",
            "label": anno["label"],
            "split": args.split,
            "packet": packet_name
        }
        packet_manifest_fp.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
        mapping_writer.writerow([original_img_path.name, new_filename, anno["label"], args.split, packet_name])
        
    packet_anno_fp.close()
    packet_manifest_fp.close()
    mapping_csv.close()
    
    # Validation
    print("Validating imported packet...")
    report = validate_packet(imported_dir, actual_samples)
    
    reports_dir = working_root / "reports" / args.language
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"{packet_name}_{args.split}_validation.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"Packet: {packet_name}\n")
        f.write(f"Split: {args.split}\n\n")
        f.write(f"Expected samples: {report['expected']}\n")
        f.write(f"Images copied: {report['images_copied']}\n")
        f.write(f"Annotations: {report['annotations']}\n")
        f.write(f"Missing images: {report['missing_images']}\n")
        f.write(f"Missing labels: {report['missing_labels']}\n")
        f.write(f"Invalid images: {report['invalid_images']}\n")
        f.write(f"Duplicate names: {report['duplicate_names']}\n\n")
        f.write(f"STATUS: {report['status']}\n")
        
    print(f"Validation complete. Status: {report['status']}")
    
    # Update registry
    registry[args.split][packet_name] = {
        "status": "imported" if report["status"] == "PASS" else "import_failed",
        "samples": report["images_copied"],
        "trained": False
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)
        
    print(f"Import of {packet_name} complete.")

if __name__ == "__main__":
    main()
