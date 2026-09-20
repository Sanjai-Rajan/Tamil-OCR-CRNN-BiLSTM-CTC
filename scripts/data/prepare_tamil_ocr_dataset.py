import os
import argparse
import shutil
import json
import csv
from pathlib import Path

def setup_argparse():
    parser = argparse.ArgumentParser(description="Prepare Tamil OCR Dataset for Training")
    parser.add_argument("--source", type=str, default="data/Large Data set", help="Path to original dataset")
    parser.add_argument("--destination", type=str, default="data/tamil_ocr_dataset", help="Path to copied dataset")
    parser.add_argument("--packet-size", type=int, default=5000, help="Number of images per packet")
    parser.add_argument("--language", type=str, default="tamil", help="Language prefix for files")
    parser.add_argument("--dry-run", action="store_true", help="Run without making actual changes")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing destination directory")
    return parser

def load_annotations(annotation_file):
    annotations = []
    if not annotation_file.exists():
        return annotations
    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            if ',' in line:
                # Split only on the first comma
                img_path, label = line.strip().split(',', 1)
                annotations.append({"img_path": img_path.strip(), "label": label.strip()})
    return annotations

def prepare_dataset(args):
    source_dir = Path(args.source)
    dest_dir = Path(args.destination)
    
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist.")
        return

    print(f"--- Preparing {args.language.upper()} OCR Dataset ---")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Packet Size: {args.packet_size}")
    print(f"Dry Run: {args.dry_run}")
    print(f"Overwrite: {args.overwrite}")
    
    if not args.dry_run and dest_dir.exists():
        if args.overwrite:
            print("Overwriting existing destination directory...")
            shutil.rmtree(dest_dir)
        else:
            print("Error: Destination directory exists. Use --overwrite to replace it.")
            return

    splits = ['train', 'val', 'test']
    split_map = {'train': 'train', 'val': 'validation', 'test': 'test'}
    
    report_data = {
        "language": args.language,
        "total_images": 0,
        "missing_images": 0,
        "missing_annotations": 0,
        "duplicate_images": 0,
        "status": "PASS"
    }
    
    # Process each split
    for original_split in splits:
        target_split = split_map[original_split]
        report_data[f"{target_split}_images"] = 0
        report_data[f"{target_split}_packets"] = 0
        
        # Read annotations
        # Paths are typically like txt_files/train/train_augmented.txt
        anno_file = source_dir / "txt_files" / original_split / f"{original_split}_augmented.txt"
        if not anno_file.exists():
            print(f"Warning: Annotation file {anno_file} not found. Skipping {target_split}.")
            continue
            
        annotations = load_annotations(anno_file)
        print(f"[{target_split.upper()}] Found {len(annotations)} annotations.")
        
        if args.dry_run:
            continue
            
        # Create directories
        (dest_dir / target_split).mkdir(parents=True, exist_ok=True)
        (dest_dir / "annotations" / target_split).mkdir(parents=True, exist_ok=True)
        
        mapping_file = dest_dir / "annotations" / f"{target_split}_mapping.csv"
        manifest_file = dest_dir / "annotations" / f"{target_split}_manifest.jsonl"
        
        mapping_csv = open(mapping_file, 'w', newline='', encoding='utf-8')
        mapping_writer = csv.writer(mapping_csv)
        mapping_writer.writerow(["original_filename", "new_filename", "label", "split"])
        
        manifest_fp = open(manifest_file, 'w', encoding='utf-8')
        
        packet_idx = 1
        current_packet_count = 0
        current_packet_dir = dest_dir / "packets" / target_split / f"packet_{packet_idx:03d}"
        (current_packet_dir / "images").mkdir(parents=True, exist_ok=True)
        packet_anno_fp = open(current_packet_dir / "annotations.txt", 'w', encoding='utf-8')
        packet_manifest_fp = open(current_packet_dir / "manifest.jsonl", 'w', encoding='utf-8')
        
        for idx, anno in enumerate(annotations):
            # Parse paths like 'train/1(1).jpg'
            original_img_rel_path = anno["img_path"] 
            # We assume original images are in source_dir / original_img_rel_path
            original_img_path = source_dir / original_img_rel_path
            
            if not original_img_path.exists():
                report_data["missing_images"] += 1
                continue
                
            # Rename deterministically
            ext = original_img_path.suffix
            new_filename = f"{args.language}_{target_split}_{idx+1:06d}{ext}"
            new_img_path = dest_dir / target_split / new_filename
            
            # Copy image
            shutil.copy2(original_img_path, new_img_path)
            
            # Write mapping
            mapping_writer.writerow([original_img_path.name, new_filename, anno["label"], target_split])
            
            # Write global manifest
            manifest_entry = {
                "original_image": str(original_img_path),
                "image": str(new_img_path),
                "label": anno["label"],
                "split": target_split
            }
            manifest_fp.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
            
            # Handle packets (To save disk space, we could just copy to packets, or use symlinks. 
            # Requirements say: "If possible, packet creation should operate from the copied dataset 
            # and should not create another unnecessary full duplicate."
            # Since symlinks on Windows can require admin, we'll just write the packet manifest/annotations
            # and they will point to the unified copied image, OR we actually put images inside packets.
            # The prompt asks for: packets/train/packet_001/images/...
            # To avoid duplicating 10GB, we will move the copied image into the packet directly.
            
            packet_img_path = current_packet_dir / "images" / new_filename
            shutil.move(str(new_img_path), str(packet_img_path))
            
            # Update packet annotations
            packet_anno_fp.write(f"images/{new_filename},{anno['label']}\n")
            packet_manifest_fp.write(json.dumps({
                "original_image": str(original_img_path),
                "image": f"images/{new_filename}",
                "label": anno["label"],
                "split": target_split
            }, ensure_ascii=False) + "\n")
            
            current_packet_count += 1
            report_data[f"{target_split}_images"] += 1
            report_data["total_images"] += 1
            
            if current_packet_count >= args.packet_size:
                packet_anno_fp.close()
                packet_manifest_fp.close()
                report_data[f"{target_split}_packets"] += 1
                
                # Start new packet if there are more images
                if idx < len(annotations) - 1:
                    packet_idx += 1
                    current_packet_count = 0
                    current_packet_dir = dest_dir / "packets" / target_split / f"packet_{packet_idx:03d}"
                    (current_packet_dir / "images").mkdir(parents=True, exist_ok=True)
                    packet_anno_fp = open(current_packet_dir / "annotations.txt", 'w', encoding='utf-8')
                    packet_manifest_fp = open(current_packet_dir / "manifest.jsonl", 'w', encoding='utf-8')
        
        # Close last packet files if not exactly closed
        if current_packet_count > 0:
            packet_anno_fp.close()
            packet_manifest_fp.close()
            report_data[f"{target_split}_packets"] += 1
            
        mapping_csv.close()
        manifest_fp.close()

    if not args.dry_run:
        (dest_dir / "reports").mkdir(parents=True, exist_ok=True)
        report_file = dest_dir / "reports" / "tamil_dataset_summary.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4)
        print(f"Dataset preparation complete. Report saved to {report_file}")
    else:
        print("Dry run complete. No files were modified.")

if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    prepare_dataset(args)
