import os
import json
import shutil
import random
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DS3_SOURCE = (PROJECT_ROOT.parent.parent / "DS UAR/3")
DEST_ROOT = PROJECT_ROOT / "data" / "train_ready" / "tamil" / "character_classification"
MANIFEST_ROOT = PROJECT_ROOT / "data" / "manifests" / "tamil" / "character_classification"
VOCAB_PATH = PROJECT_ROOT / "data" / "tamil_ocr_dataset" / "vocabulary" / "tamil_vocab.json"
OUT_REPORT = PROJECT_ROOT / "outputs" / "dataset_source_audit" / "DS3_FINAL_INTEGRATION_REPORT.md"

def load_vocab():
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get('characters', []))
    return set()

vocab_chars = load_vocab()

# TASK 2: REVIEW THE 19 PHONETIC LABELS
mapping_dict = {
    "ai": "ஐ",
    "cha": "ச",
    "ee": "ஈ",
    "la": "ல",
    "ma": "ம",
    "moo": "மூ",
    "nna": "REQUIRES_REVIEW",   # ண or ன? Ambiguous.
    "nnna": "REQUIRES_REVIEW",  # ண or ன? Ambiguous.
    "nu": "REQUIRES_REVIEW",    # நு, ணு, or னு? Ambiguous.
    "nuu": "REQUIRES_REVIEW",   # நூ, ணூ, or னூ? Ambiguous.
    "oo": "REQUIRES_REVIEW",    # ஒ or ஓ or ஊ? Ambiguous.
    "pa": "ப",
    "ra": "REQUIRES_REVIEW",    # ர or ற? Ambiguous.
    "tha": "த",
    "va": "வ",
    "vee": "வீ",
    "vu": "வு",
    "y": "ய்",
    "ya": "ய",
    "zha": "ழ"
}

# Write mapping file
mapping_file = MANIFEST_ROOT / "dataset_03_label_mapping.json"
MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
with open(mapping_file, 'w', encoding='utf-8') as f:
    json.dump(mapping_dict, f, ensure_ascii=False, indent=4)

def run_integration():
    dest_dir = DEST_ROOT / "dataset_03"
    
    # Clean previous integration copy
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Gather source images recursively to recover 'vee/vu' and others
    # Key is the original class name (folder name). For nested, we'll extract the immediate parent folder as the class.
    class_images = {}
    for root, dirs, files in os.walk(DS3_SOURCE):
        root_path = Path(root)
        if root_path == DS3_SOURCE:
            continue
        
        orig_class = root_path.name
        if orig_class not in class_images:
            class_images[orig_class] = []
            
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in {'.png', '.jpg', '.jpeg', '.tif', '.bmp'}:
                class_images[orig_class].append(root_path / file)

    manifest_entries = []
    classes_info = []
    
    report_data = {
        "integrated_count_prev": 2465,
        "recovered_count": 85,
        "images": 0,
        "train": 0,
        "val": 0,
        "test": 0,
        "oov_before": 19,
        "oov_after": 0,
        "errors": 0
    }
    
    class_id_counter = 0
    
    # Sort for determinism
    for orig_class in sorted(class_images.keys()):
        images = sorted(class_images[orig_class])
        if not images:
            continue
            
        # Map label
        mapped_label = mapping_dict.get(orig_class, "REQUIRES_REVIEW")
        
        # If ambiguous or truly OOV, keep it as OOV
        is_oov = False
        final_label = mapped_label
        
        if mapped_label == "REQUIRES_REVIEW":
            is_oov = True
            report_data["oov_after"] += 1
            final_label = orig_class # fallback to original english name for folder
        else:
            # Check characters against vocab
            for char in mapped_label:
                if char not in vocab_chars:
                    is_oov = True
                    report_data["oov_after"] += 1
                    break
        
        dest_class_dir = dest_dir / final_label
        dest_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Split (80/10/10)
        random.seed(42 + class_id_counter)
        n = len(images)
        random.shuffle(images)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        valid_img_count = 0
        
        for i, img_path in enumerate(images):
            # Validate
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    if w == 0 or h == 0:
                        raise ValueError("Zero dimensions")
            except:
                report_data["errors"] += 1
                continue
                
            if i < n_train:
                split = "train"
                report_data["train"] += 1
            elif i < n_train + n_val:
                split = "val"
                report_data["val"] += 1
            else:
                split = "test"
                report_data["test"] += 1
                
            dest_img_path = dest_class_dir / img_path.name
            shutil.copy2(img_path, dest_img_path)
            
            valid_img_count += 1
            report_data["images"] += 1
            
            # Forward slashes for manifest
            rel_path = f"dataset_03/{final_label}/{img_path.name}"
            
            manifest_entries.append({
                "image": rel_path,
                "label": final_label if not is_oov else orig_class,
                "dataset": "dataset_03",
                "class_id": class_id_counter,
                "split": split,
                "original_label": orig_class,
                "mapped_label": mapped_label
            })
            
        classes_info.append({
            "class_id": class_id_counter,
            "class_label": final_label if not is_oov else orig_class,
            "original_label": orig_class,
            "mapped_label": mapped_label,
            "unicode_hex": " ".join([f"U+{ord(c):04X}" for c in (final_label if not is_oov else orig_class)]),
            "image_count": valid_img_count,
            "is_oov": is_oov
        })
        
        class_id_counter += 1

    # Write manifests
    manifest_file = MANIFEST_ROOT / "dataset_03.jsonl"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    classes_file = MANIFEST_ROOT / "dataset_03_classes.json"
    with open(classes_file, 'w', encoding='utf-8') as f:
        json.dump(classes_info, f, ensure_ascii=False, indent=2)

    # Output Report
    md = f"""# DS3 Final Integration Report

- **Original Image Count (DS UAR\\3 Total)**: 2550
- **Previously Integrated Count**: {report_data['integrated_count_prev']}
- **Recovered Count**: {report_data['images'] - report_data['integrated_count_prev']}
- **Final Valid Integrated Count**: {report_data['images']}
- **Total Classes**: {class_id_counter}
- **OOV Before Mapping**: {report_data['oov_before']}
- **OOV After Mapping**: {report_data['oov_after']}
- **Excluded Files (Errors)**: {report_data['errors']}
- **Train / Val / Test**: {report_data['train']} / {report_data['val']} / {report_data['test']}

## Status
**Final Status**: {'READY' if report_data['oov_after'] == 0 else 'READY_AFTER_MANUAL_REVIEW'}

## Label Mapping Table
"""
    md += "| Original Label | Mapped Label | Note |\n"
    md += "| :--- | :--- | :--- |\n"
    for cl in classes_info:
        note = "Ambiguous, Requires Review" if cl['mapped_label'] == "REQUIRES_REVIEW" else "Mapped to Unicode"
        md += f"| {cl['original_label']} | {cl['mapped_label']} | {note} |\n"

    with open(OUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(md)
        
    print(f"Dataset 3:")
    print(f"STATUS = {'READY' if report_data['oov_after'] == 0 else 'READY_AFTER_MANUAL_REVIEW'}")

if __name__ == "__main__":
    run_integration()
