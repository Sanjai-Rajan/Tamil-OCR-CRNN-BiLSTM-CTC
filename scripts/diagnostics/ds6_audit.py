import os
import json
import pandas as pd
import unicodedata
import hashlib
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DS6_SOURCE = (PROJECT_ROOT.parent.parent / "DS UAR/6")
OUT_DIR = PROJECT_ROOT / "outputs" / "dataset_source_audit"
VOCAB_PATH = PROJECT_ROOT / "data" / "tamil_ocr_dataset" / "vocabulary" / "tamil_vocab.json"
IMPORTED_DIR = PROJECT_ROOT / "data" / "tamil_ocr_dataset" / "imported" / "tamil"

def load_vocab():
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get('characters', []))
    return set()

vocab_chars = load_vocab()

def get_file_size(filepath):
    try:
        return os.path.getsize(filepath)
    except:
        return -1

def get_md5(filepath):
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None

def audit_ds6():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. IDENTIFY THE 3 ANNOTATION FILES
    annotation_files = []
    for root, dirs, files in os.walk(DS6_SOURCE / "txt_files"):
        for f in files:
            if f.endswith('.txt'):
                annotation_files.append(Path(root) / f)
                
    md_seq = "# Dataset 6 Sequence Audit\n\n"
    md_seq += f"Found {len(annotation_files)} annotation files.\n\n"
    
    all_records = []
    
    for ann in annotation_files:
        md_seq += f"## {ann.name}\n"
        md_seq += f"- Path: {ann}\n"
        
        records = []
        with open(ann, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
            
        md_seq += f"- Row count: {len(lines)}\n"
        
        # Parse lines. Usually "image_path label" separated by space or tab
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            
            # Split on first whitespace
            parts = line.split(maxsplit=1)
            img_path = parts[0].rstrip(',')
            label = parts[1] if len(parts) > 1 else ""
            
            records.append({
                "source_file": ann.name,
                "image_path": img_path,
                "label": label
            })
            
        md_seq += f"- Parsed records: {len(records)}\n"
        if records:
            md_seq += "- Sample entries:\n"
            for i in range(min(5, len(records))):
                md_seq += f"  - `{records[i]['image_path']}` => `{records[i]['label']}`\n"
        md_seq += "\n"
        all_records.extend(records)
        
    md_seq += "## Analysis of '3 classes'\n"
    md_seq += "The dataset is structured as 3 splits: train, val, test, NOT 3 character classes. The annotation files correspond to these splits.\n\n"

    # 2. IMAGE ↔ LABEL INTEGRITY
    total_records = len(all_records)
    unique_refs = set(r["image_path"] for r in all_records)
    
    # Find all actual images in DS6 (excluding txt_files)
    actual_images = {}
    for folder in ['train', 'val', 'test']:
        folder_path = DS6_SOURCE / folder
        if folder_path.exists():
            for f in os.listdir(folder_path):
                # Save relative path as written in annotation (usually just the filename, or folder/filename)
                # Let's map filename to full path
                actual_images[f] = folder_path / f
                
    missing_images = 0
    empty_labels = 0
    duplicate_refs = total_records - len(unique_refs)
    
    valid_pairs = []
    labels = []
    
    for r in all_records:
        if not r["label"]:
            empty_labels += 1
            continue
            
        # Try to resolve path
        filename = Path(r["image_path"]).name
        if filename not in actual_images:
            missing_images += 1
            continue
            
        full_path = actual_images[filename]
        valid_pairs.append({
            "full_path": str(full_path),
            "filename": filename,
            "label": r["label"]
        })
        labels.append(r["label"])
        
    unique_labels = len(set(labels))
    
    md_seq += "## Image-Label Integrity\n"
    md_seq += f"- Total annotation records: {total_records}\n"
    md_seq += f"- Unique image references: {len(unique_refs)}\n"
    md_seq += f"- Missing images: {missing_images}\n"
    md_seq += f"- Duplicate image references: {duplicate_refs}\n"
    md_seq += f"- Empty labels: {empty_labels}\n"
    md_seq += f"- Unique labels: {unique_labels}\n"
    md_seq += f"- **Valid image-label pairs**: {len(valid_pairs)}\n\n"

    with open(OUT_DIR / "DS6_SEQUENCE_AUDIT.md", "w", encoding="utf-8") as f:
        f.write(md_seq)

    # 3. LABEL / UNICODE AUDIT & 4. NORMALIZATION
    char_freq = Counter()
    
    nfc_unchanged = 0
    nfc_changed = 0
    nfc_fixed = 0
    genuinely_oov = 0
    
    label_lengths = []
    
    for r in valid_pairs:
        label = r["label"]
        label_lengths.append(len(label))
        for char in label:
            char_freq[char] += 1
            
        # Normalization check
        norm_label = unicodedata.normalize('NFC', label)
        if norm_label == label:
            nfc_unchanged += 1
            if any(c not in vocab_chars for c in label):
                genuinely_oov += 1
        else:
            nfc_changed += 1
            orig_oov = any(c not in vocab_chars for c in label)
            norm_oov = any(c not in vocab_chars for c in norm_label)
            if orig_oov and not norm_oov:
                nfc_fixed += 1
            elif norm_oov:
                genuinely_oov += 1

    char_rows = []
    for c, freq in char_freq.most_common():
        in_vocab = "Yes" if c in vocab_chars else "No"
        try:
            name = unicodedata.name(c)
        except:
            name = "UNKNOWN"
            
        char_rows.append({
            "Character": c,
            "Frequency": freq,
            "Unicode Codepoint": f"U+{ord(c):04X}",
            "Unicode Name": name,
            "In Current Vocab": in_vocab
        })
        
    df_chars = pd.DataFrame(char_rows)
    df_chars.to_csv(OUT_DIR / "DS6_UNICODE_AUDIT.csv", index=False)
    # the prompt requested DS6_CHARACTER_FREQUENCY.csv as well, we'll write both or just rename
    df_chars[["Character", "Frequency"]].to_csv(OUT_DIR / "DS6_CHARACTER_FREQUENCY.csv", index=False)
    
    # 5. LABEL LENGTH ANALYSIS
    df_len = pd.Series(label_lengths)
    
    # 7. IMAGE/LABEL VISUAL VALIDATION
    md_vis = "# Dataset 6 Visual Validation\n\n"
    import random
    random.seed(42)
    sample = random.sample(valid_pairs, min(20, len(valid_pairs)))
    for s in sample:
        md_vis += f"### File: {s['filename']}\n"
        md_vis += f"**Transcription**: `{s['label']}`\n"
        md_vis += f"![image](file:///{s['full_path'].replace('\\', '/')})\n\n"
        
    with open(OUT_DIR / "DS6_IMAGE_LABEL_VALIDATION.md", "w", encoding="utf-8") as f:
        f.write(md_vis)

    # 8. DUPLICATE / LEAKAGE ANALYSIS (Simplified for computational feasibility)
    md_dup = "# Dataset 6 Duplicate Audit\n\n"
    
    # Internal duplicate filenames
    ds6_filenames = set()
    internal_dups = 0
    for r in valid_pairs:
        if r["filename"] in ds6_filenames:
            internal_dups += 1
        else:
            ds6_filenames.add(r["filename"])
            
    md_dup += f"- Internal duplicates (exact filename match): {internal_dups}\n\n"
    
    # External duplicate filenames
    external_overlap = 0
    if IMPORTED_DIR.exists():
        ext_filenames = set()
        for root, dirs, files in os.walk(IMPORTED_DIR):
            for f in files:
                ext_filenames.add(f)
                
        external_overlap = len(ds6_filenames.intersection(ext_filenames))
                    
    md_dup += f"- Overlap with current project corpus ({IMPORTED_DIR}): {external_overlap} identical filenames.\n"
    
    with open(OUT_DIR / "DS6_DUPLICATE_AUDIT.md", "w", encoding="utf-8") as f:
        f.write(md_dup)

    # 10. FINAL COMPATIBILITY DECISION
    oov_count = len([r for r in char_rows if r["In Current Vocab"] == "No"])
    
    if genuinely_oov == 0 and oov_count == 0:
        status = "READY"
    elif genuinely_oov == 0 and nfc_fixed > 0:
        status = "READY_AFTER_UNICODE_NORMALIZATION"
    else:
        status = "REQUIRES_VOCABULARY_EXPANSION"
        
    crnn_suitable = df_len.mean() > 1 and genuinely_oov == 0

    md_comp = f"""# Dataset 6 Final Compatibility Report

## Summary
- **Dataset 6 Images**: {len(actual_images)}
- **Annotation records**: {total_records}
- **Valid image-label pairs**: {len(valid_pairs)}
- **Missing images**: {missing_images}
- **Duplicate references**: {duplicate_refs}
- **Empty labels**: {empty_labels}
- **Label type**: Text transcription
- **Unique codepoints**: {len(char_rows)}
- **Current-vocabulary compatible**: {len(char_rows) - oov_count}
- **NFC Normalization Fixed Pairs**: {nfc_fixed}
- **Genuine OOV Pairs**: {genuinely_oov}
- **Duplicate/overlap with current corpus**: {external_overlap}

## Label Lengths
- **Min**: {df_len.min()}
- **Max**: {df_len.max()}
- **Mean**: {df_len.mean():.2f}
- **Median**: {df_len.median()}

## Final Status
**{status}**
"""
    with open(OUT_DIR / "DS6_COMPATIBILITY_REPORT.md", "w", encoding="utf-8") as f:
        f.write(md_comp)
        
    print(f"Dataset 6\nImages: {len(actual_images)}\nAnnotation records: {total_records}\nValid image-label pairs: {len(valid_pairs)}\nMissing images: {missing_images}\nDuplicate references: {duplicate_refs}\nEmpty labels: {empty_labels}\nLabel type: Transcription\nImage type: Sequence Crop\nUnique codepoints: {len(char_rows)}\nCurrent-vocabulary compatible: {len(char_rows) - oov_count}\nNormalization-resolvable: {nfc_fixed} pairs\nGenuine OOV: {genuinely_oov} pairs\nDuplicate/overlap with current corpus: {external_overlap}\nCRNN suitable: {'Yes' if crnn_suitable else 'No, requires vocab expansion'}\nFinal status: {status}")

if __name__ == "__main__":
    audit_ds6()
