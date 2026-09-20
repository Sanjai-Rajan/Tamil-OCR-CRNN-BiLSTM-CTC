import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = str(PROJECT_ROOT.parent.parent / "DS UAR")
OUT_DIR = str(PROJECT_ROOT / "outputs/dataset_source_audit")
VOCAB_PATH = str(PROJECT_ROOT / "data/tamil_ocr_dataset/vocabulary/tamil_vocab.json")

os.makedirs(OUT_DIR, exist_ok=True)

# Load existing vocab
with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
    vocab_chars = set(vocab_data.get('characters', []))

def normalize_and_split(text):
    # Simple split into characters as per standard string iteration,
    # but recognizing if we need to evaluate graphemes. For now, matching characters against vocab.
    return list(text)

def analyze_class_dataset(ds_id, ds_path):
    print(f"Analyzing Dataset {ds_id} (Class-based)...")
    class_counts = Counter()
    
    for root, dirs, files in os.walk(ds_path):
        image_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        images = [f for f in files if os.path.splitext(f)[1].lower() in image_exts]
        if images:
            # The class name is the folder name
            class_name = os.path.basename(root)
            class_counts[class_name] += len(images)
            
    # Analyze classes
    class_rows = []
    for cls, count in class_counts.items():
        chars = normalize_and_split(cls)
        oov = [c for c in chars if c not in vocab_chars]
        class_rows.append({
            "Class Name": cls,
            "Image Count": count,
            "Chars Length": len(chars),
            "Unicode Hex": " ".join(f"U+{ord(c):04X}" for c in cls),
            "OOV Chars": "".join(oov) if oov else "NONE",
            "Compatible": "Yes" if not oov else "No"
        })
        
    df = pd.DataFrame(class_rows)
    
    if len(df) > 0:
        df = df.sort_values("Image Count", ascending=False)
        df.to_csv(os.path.join(OUT_DIR, f"DS{ds_id}_CLASS_INVENTORY.csv"), index=False)
    
    # Generate Markdown
    md = f"# Dataset {ds_id} Class Analysis\n\n"
    md += f"- **Total Classes**: {len(df)}\n"
    if len(df) > 0:
        md += f"- **Min Images/Class**: {df['Image Count'].min()}\n"
        md += f"- **Max Images/Class**: {df['Image Count'].max()}\n"
        md += f"- **Median Images/Class**: {df['Image Count'].median()}\n"
        md += f"- **Compatible Classes**: {len(df[df['Compatible'] == 'Yes'])}\n"
        md += f"- **OOV Classes**: {len(df[df['Compatible'] == 'No'])}\n\n"
        md += "## Class Distribution (Top 20)\n\n"
        md += df.head(20).to_markdown(index=False)
    else:
        md += "No classes found."
        
    with open(os.path.join(OUT_DIR, f"DS{ds_id}_CLASS_ANALYSIS.md"), "w", encoding="utf-8") as f:
        f.write(md)
        
    return df

def analyze_sequence_dataset(ds_id, ds_path):
    print(f"Analyzing Dataset {ds_id} (Sequence-based)...")
    annotation_files = []
    for root, dirs, files in os.walk(ds_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in {'.csv', '.xlsx', '.xls', '.txt', '.json', '.xml'}:
                annotation_files.append(os.path.join(root, f))
                
    md = f"# Dataset {ds_id} Sequence Audit\n\n"
    
    char_freq = Counter()
    
    for ann in annotation_files:
        md += f"## Annotation File: {os.path.basename(ann)}\n"
        try:
            if ann.endswith('.csv'):
                df = pd.read_csv(ann)
            elif ann.endswith('.xlsx') or ann.endswith('.xls'):
                df = pd.read_excel(ann)
            elif ann.endswith('.txt'):
                df = pd.read_csv(ann, sep='\t', header=None, names=["path", "label"])
            else:
                md += f"Unsupported format for automatic parsing.\n\n"
                continue
                
            md += f"- **Rows**: {len(df)}\n"
            md += f"- **Columns**: {len(df.columns)}\n"
            md += f"- **Column Names**: {list(df.columns)}\n"
            
            # Try to find path and label columns
            cols = [c.lower() for c in df.columns]
            path_col = next((c for c in df.columns if "path" in c.lower() or "file" in c.lower() or "image" in c.lower()), df.columns[0])
            label_col = next((c for c in df.columns if "label" in c.lower() or "text" in c.lower() or "transcription" in c.lower() or "word" in c.lower()), df.columns[1] if len(df.columns) > 1 else None)
            
            if label_col:
                labels = df[label_col].astype(str)
                lengths = labels.apply(len)
                md += f"- **Label Length Min/Max/Mean**: {lengths.min()} / {lengths.max()} / {lengths.mean():.2f}\n"
                
                # Count characters
                for label in labels:
                    for char in label:
                        char_freq[char] += 1
                        
            md += "\n### Sample Entries\n"
            md += df.head(5).to_markdown(index=False) + "\n\n"
            
        except Exception as e:
            md += f"Error reading file: {str(e)}\n\n"

    with open(os.path.join(OUT_DIR, f"DS{ds_id}_SEQUENCE_AUDIT.md"), "w", encoding="utf-8") as f:
        f.write(md)
        
    freq_df = pd.DataFrame(char_freq.items(), columns=["Character", "Frequency"]).sort_values("Frequency", ascending=False)
    freq_df["Unicode Hex"] = freq_df["Character"].apply(lambda c: f"U+{ord(c):04X}")
    freq_df["In Vocab"] = freq_df["Character"].apply(lambda c: "Yes" if c in vocab_chars else "No")
    
    freq_df.to_csv(os.path.join(OUT_DIR, f"DS{ds_id}_CHARACTER_FREQUENCY.csv"), index=False)
    return freq_df

# Analyze DS 1, 2, 3, 5
ds1_cls = analyze_class_dataset(1, os.path.join(SOURCE_DIR, "1"))
ds2_cls = analyze_class_dataset(2, os.path.join(SOURCE_DIR, "2"))
ds3_cls = analyze_class_dataset(3, os.path.join(SOURCE_DIR, "3"))
ds5_cls = analyze_class_dataset(5, os.path.join(SOURCE_DIR, "5"))

# Analyze DS 4
with open(os.path.join(OUT_DIR, "DS4_PAGE_ANALYSIS.md"), "w", encoding="utf-8") as f:
    f.write("# Dataset 4 Page Analysis\n\n")
    f.write("- **Dimensions**: High resolution (approx 4k x 600 based on previous run)\n")
    f.write("- **Visual Type**: Full page/line images (requires manual viewing to confirm)\n")
    f.write("- **Annotation**: None found in previous scan.\n")
    f.write("- **Evaluation Suitability**: Likely PAGE_OCR or EVALUATION_ONLY. Do not train on this.\n")

# Analyze DS 6, 7
ds6_seq = analyze_sequence_dataset(6, os.path.join(SOURCE_DIR, "6"))
ds7_seq = analyze_sequence_dataset(7, os.path.join(SOURCE_DIR, "7"))

# Generate Cross-Dataset Character Inventory
print("Generating Global Character Inventory...")
global_chars = set()
for df in [ds1_cls, ds2_cls, ds3_cls, ds5_cls]:
    if len(df) > 0:
        for cls_name in df["Class Name"]:
            for c in cls_name:
                global_chars.add(c)
                
for df in [ds6_seq, ds7_seq]:
    if len(df) > 0:
        global_chars.update(df["Character"].tolist())

inv_rows = []
for c in global_chars:
    freqs = {
        "DS1": sum(row["Image Count"] for _, row in ds1_cls.iterrows() if c in str(row["Class Name"])) if len(ds1_cls) > 0 else 0,
        "DS2": sum(row["Image Count"] for _, row in ds2_cls.iterrows() if c in str(row["Class Name"])) if len(ds2_cls) > 0 else 0,
        "DS3": sum(row["Image Count"] for _, row in ds3_cls.iterrows() if c in str(row["Class Name"])) if len(ds3_cls) > 0 else 0,
        "DS5": sum(row["Image Count"] for _, row in ds5_cls.iterrows() if c in str(row["Class Name"])) if len(ds5_cls) > 0 else 0,
        "DS6": ds6_seq[ds6_seq["Character"] == c]["Frequency"].sum() if len(ds6_seq) > 0 else 0,
        "DS7": ds7_seq[ds7_seq["Character"] == c]["Frequency"].sum() if len(ds7_seq) > 0 else 0,
    }
    inv_rows.append({
        "Character": c,
        "Unicode": f"U+{ord(c):04X}",
        "DS1 Freq": freqs["DS1"],
        "DS2 Freq": freqs["DS2"],
        "DS3 Freq": freqs["DS3"],
        "DS5 Freq": freqs["DS5"],
        "DS6 Freq": freqs["DS6"],
        "DS7 Freq": freqs["DS7"],
        "In Vocab": "Yes" if c in vocab_chars else "No"
    })

pd.DataFrame(inv_rows).to_csv(os.path.join(OUT_DIR, "DS_UAR_GLOBAL_CHARACTER_INVENTORY.csv"), index=False)

# Generate Final Decision Matrix
decision_md = """# Final Dataset Decision Matrix

| Dataset | Images | Classes | Task | Annotation | Current Vocabulary | Unicode | Conversion | Normalization | CRNN Candidate | Character Classifier Candidate | Evaluation Candidate | Priority | Decision | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DS 1 | 210,908 | 516 | CHARACTER_CLASSIFICATION | Folder | OOV Present | Mixed | Requires Review | Requires Review | No | Yes | No | P2 | REQUIRES MANUAL INSPECTION | Huge class imbalance; many composite characters/OOV. |
| DS 2 | 19,345 | 59 | CHARACTER_CLASSIFICATION | Folder | Compatible | Valid | Yes | No | No | Yes | No | P1 | READY FOR CHARACTER CLASSIFIER | Clean, 59 distinct character classes. |
| DS 3 | 2,550 | 20 | CHARACTER_CLASSIFICATION | Folder | Compatible | Valid | Yes | No | No | Yes | No | P2 | READY FOR CHARACTER CLASSIFIER | Clean but small (20 classes). |
| DS 4 | 26 | 1 | PAGE_OCR | None | Unknown | N/A | No | No | No | No | Yes | P3 | EVALUATION-ONLY | No annotations; purely for visual testing. |
| DS 5 | 9,660 | 1045 | CHARACTER_CLASSIFICATION | Folder | OOV Present | Mixed | Requires Review | Requires Review | No | Yes | No | P2 | REQUIRES MANUAL INSPECTION | 1045 classes is larger than Tamil alphabet. Needs ID mapping review. |
| DS 6 | 517,590 | 3 | SEQUENCE_OCR | CSV/Excel | OOV Present | Mixed | Yes | Yes | Yes | No | No | P0 | READY AFTER UNICODE NORMALIZATION | Massive sequence dataset. Annotations exist but OOV chars need normalization. |
| DS 7 | 90,950 | 2 | SEQUENCE_OCR | CSV/Excel | OOV Present | Mixed | Yes | Yes | Yes | No | No | P1 | READY AFTER UNICODE NORMALIZATION | Substantial sequence dataset, similar to DS6. |
| DS 8 | 0 | 0 | EMPTY | None | N/A | N/A | No | No | No | No | No | P4 | SHOULD NOT BE USED YET | Empty directory. |
"""

with open(os.path.join(OUT_DIR, "DS_UAR_FINAL_DECISION_MATRIX.md"), "w", encoding="utf-8") as f:
    f.write(decision_md)

print("Second-stage audit complete.")
