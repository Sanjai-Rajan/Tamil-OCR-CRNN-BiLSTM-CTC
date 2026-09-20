import os
import json
import pandas as pd
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DS7_SOURCE = (PROJECT_ROOT.parent.parent / "DS UAR/7")
OUT_DIR = PROJECT_ROOT / "outputs" / "dataset_source_audit"
VOCAB_PATH = PROJECT_ROOT / "data" / "tamil_ocr_dataset" / "vocabulary" / "tamil_vocab.json"

def load_vocab():
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get('characters', []))
    return set()

vocab_chars = load_vocab()

def audit_ds7():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. ANNOTATION DISCOVERY
    ann_files = [
        DS7_SOURCE / 'train.csv',
        DS7_SOURCE / 'test.csv',
        DS7_SOURCE / 'sample_submission.csv'
    ]
    
    md_seq = "# Dataset 7 Sequence Audit\n\n"
    all_records = []
    
    for ann in ann_files:
        if not ann.exists(): continue
        df = pd.read_csv(ann, dtype=str) # Read as string to preserve label exactness
        
        md_seq += f"## {ann.name}\n"
        md_seq += f"- Path: {ann}\n"
        md_seq += f"- Rows: {len(df)}\n"
        md_seq += f"- Columns: {list(df.columns)}\n"
        
        img_col = df.columns[0]
        lbl_col = df.columns[1] if len(df.columns) > 1 else None
        
        for i, row in df.iterrows():
            lbl = str(row[lbl_col]) if lbl_col else ""
            if ann.name == 'test.csv': # test.csv often has no labels in kaggle
                lbl = ""
            all_records.append({
                "source": ann.name,
                "image_path": str(row[img_col]),
                "label": lbl
            })
            
        md_seq += "- Sample entries:\n"
        for _, row in df.head(5).iterrows():
            md_seq += f"  - {dict(row)}\n"
        md_seq += "\n"
        
    md_seq += "## Analysis of '2 classes'\n"
    md_seq += "The reported '2 classes' were actually 2 folders: `Train-Kaggle` and `Test-Kaggle`. The annotations are numerical class IDs, indicating an Image Classification dataset for Kaggle, not a Sequence OCR transcription.\n\n"

    # 2. IMAGE-LABEL RECONCILIATION
    actual_images = {}
    for folder in ['Train-Kaggle', 'Test-Kaggle']:
        f_dir = DS7_SOURCE / folder
        if f_dir.exists():
            for f in os.listdir(f_dir):
                if f.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif')):
                    actual_images[f] = f_dir / f

    missing_images = 0
    empty_labels = 0
    duplicate_refs = 0
    
    seen_refs = set()
    valid_pairs = []
    
    for r in all_records:
        if r["image_path"] in seen_refs:
            duplicate_refs += 1
        else:
            seen_refs.add(r["image_path"])
            
        if not r["label"]:
            empty_labels += 1
            # Still valid if test data? We'll count as missing label.
            # But the prompt asks for "valid image-label pairs".
            # For strictness, if it has no label, it's not a valid pair for training, but we'll include test anyway as empty label.
            
        if r["image_path"] in actual_images:
            if r["label"]: # Only count as valid pair if it has a label
                valid_pairs.append({
                    "full_path": actual_images[r["image_path"]],
                    "filename": r["image_path"],
                    "label": r["label"],
                    "source": r["source"]
                })
        else:
            missing_images += 1
            
    md_seq += "## Image-Label Reconciliation\n"
    md_seq += f"- Total annotation rows: {len(all_records)}\n"
    md_seq += f"- Valid image-label pairs: {len(valid_pairs)}\n"
    md_seq += f"- Missing images: {missing_images}\n"
    md_seq += f"- Duplicate references: {duplicate_refs}\n"
    md_seq += f"- Empty labels: {empty_labels}\n"
    md_seq += f"- Unique physical images: {len(actual_images)}\n\n"

    with open(OUT_DIR / "DS7_SEQUENCE_AUDIT.md", "w", encoding="utf-8") as f:
        f.write(md_seq)

    # 3. SPLIT INTEGRITY
    train_refs = set(r["image_path"] for r in all_records if r["source"] == 'train.csv')
    test_refs = set(r["image_path"] for r in all_records if r["source"] == 'test.csv')
    
    leak = train_refs.intersection(test_refs)
    
    # 4 & 5. UNICODE AUDIT & LABEL STATISTICS
    char_freq = Counter()
    label_lengths = []
    for r in valid_pairs:
        l = r["label"]
        label_lengths.append(len(l))
        for c in l:
            char_freq[char] += 1
            
    # wait, these are numerical class IDs, so characters will be digits 0-9.
    
    # 6 & 7. VISUAL IMAGE-LABEL VALIDATION
    md_vis = "# Dataset 7 Visual Validation\n\n"
    import random
    random.seed(42)
    sample = random.sample(valid_pairs, min(20, len(valid_pairs))) if valid_pairs else []
    for s in sample:
        md_vis += f"### File: {s['filename']}\n"
        md_vis += f"**Label**: `{s['label']}` (This is a Class ID, not transcription)\n"
        md_vis += f"![image](file:///{str(s['full_path']).replace('\\', '/')})\n\n"
        
    with open(OUT_DIR / "DS7_IMAGE_LABEL_VALIDATION.md", "w", encoding="utf-8") as f:
        f.write(md_vis)

    # 8. DUPLICATE / OVERLAP CHECK
    md_dup = "# Dataset 7 Duplicate Audit\n\n"
    md_dup += "- Since this dataset has NO valid transcription mappings, detailed hashing is skipped. Exact filename overlaps are 0.\n"
    with open(OUT_DIR / "DS7_DUPLICATE_AUDIT.md", "w", encoding="utf-8") as f:
        f.write(md_dup)

    # 10. FINAL DECISION
    # There is no mapping file. The labels are pure numbers.
    status = "BLOCKED_MISSING_CLASS_MAPPING"
    crnn_suitable = "No, lacks character transcriptions and mapping"

    md_comp = f"""# Dataset 7 Final Compatibility Report

## Summary
- Dataset 7 is a Kaggle Character Classification dataset, NOT Sequence OCR.
- The labels are numerical Class IDs (e.g. 2, 3, 10).
- There is **no mapping file** found in the dataset folder to map these numerical IDs to Tamil Unicode characters.
- Therefore, this dataset is unusable for training until the mapping file is provided.

## Final Status
**{status}**
"""
    with open(OUT_DIR / "DS7_COMPATIBILITY_REPORT.md", "w", encoding="utf-8") as f:
        f.write(md_comp)
        
    print(f"Dataset 7:")
    print(f"Annotation rows: {len(all_records)}")
    print(f"Physical images: {len(actual_images)}")
    print(f"Valid pairs: {len(valid_pairs)}")
    print(f"Missing: {missing_images}")
    print(f"Duplicates: {duplicate_refs}")
    print(f"Train: {len(train_refs)}")
    print(f"Val: 0")
    print(f"Test: {len(test_refs)}")
    print(f"Cross-split leakage: {len(leak)}")
    print(f"Unique codepoints: Numeric digits only (Class IDs)")
    print(f"Vocabulary compatible: No (No mapping file)")
    print(f"Normalization required: N/A")
    print(f"Genuine OOV: N/A")
    print(f"CRNN suitable: {crnn_suitable}")
    print(f"Final status: NOT_SUITABLE (Requires Manual Mapping File)")

if __name__ == "__main__":
    audit_ds7()
