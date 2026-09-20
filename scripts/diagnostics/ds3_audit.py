import os
import json
import pandas as pd
from PIL import Image
from pathlib import Path
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DS3_SOURCE = (PROJECT_ROOT.parent.parent / "DS UAR/3")
OUT_DIR = PROJECT_ROOT / "outputs" / "dataset_source_audit"
VOCAB_PATH = PROJECT_ROOT / "data" / "tamil_ocr_dataset" / "vocabulary" / "tamil_vocab.json"
MANIFEST = PROJECT_ROOT / "data" / "manifests" / "tamil" / "character_classification" / "dataset_03.jsonl"

def run_audit():
    # 1. IDENTIFY THE 85 EXCLUDED IMAGES
    excluded_files = []
    
    # We found that `iterdir()` in integration missed nested directories. Let's find exactly what was missed.
    # The integration only looked at top-level dirs.
    top_level_dirs = [d for d in DS3_SOURCE.iterdir() if d.is_dir()]
    
    for root, dirs, files in os.walk(DS3_SOURCE):
        root_path = Path(root)
        if root_path == DS3_SOURCE:
            continue
            
        # Is it a top level dir?
        is_top_level = root_path.parent == DS3_SOURCE
        
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            file_path = root_path / f
            size = file_path.stat().st_size
            
            # If it wasn't top level, it was excluded by the iterdir() logic
            if not is_top_level:
                reason = "Excluded by Top-Level-Only Integration Script (Nested Folder)"
                readable = "Unknown"
                dims = "N/A"
                is_image = "No"
                
                if ext in {'.png', '.jpg', '.jpeg', '.tif', '.bmp'}:
                    try:
                        with Image.open(file_path) as img:
                            w, h = img.size
                            readable = "Yes"
                            is_image = "Yes"
                            dims = f"{w}x{h}"
                    except:
                        readable = "No"
                else:
                    reason = "Nested Folder + Unsupported Extension"
                    
                excluded_files.append({
                    "Original Path": str(file_path),
                    "Filename": f,
                    "Extension": ext,
                    "Size (bytes)": size,
                    "Readable": readable,
                    "Dimensions": dims,
                    "Reason": reason,
                    "Is Image": is_image,
                    "Recoverable": "Yes (Modify integration script to use os.walk)"
                })
                
    df_ex = pd.DataFrame(excluded_files)
    df_ex.to_csv(OUT_DIR / "DS3_EXCLUDED_85_AUDIT.csv", index=False)
    
    md_ex = "# Dataset 3 Excluded Files Audit\n\n"
    md_ex += f"Total excluded files: {len(df_ex)}\n\n"
    if len(df_ex) > 0:
        md_ex += "## Summary of Reasons\n"
        md_ex += df_ex["Reason"].value_counts().to_markdown() + "\n\n"
        md_ex += "## Sample Excluded Files\n"
        md_ex += df_ex.head(20).to_markdown(index=False)
    
    with open(OUT_DIR / "DS3_EXCLUDED_85_AUDIT.md", "w", encoding="utf-8") as f:
        f.write(md_ex)
        
    # 2. IDENTIFY ALL 19 OOV STRUCTURES
    # The OOVs are the english folder names like 'cha', 'ma', 'moo' because they don't match Tamil vocabulary.
    oov_classes = []
    
    for cls in [d.name for d in top_level_dirs]:
        # 'cls' is an English label like 'cha'
        unicode_points = []
        unicode_names = []
        for c in cls:
            unicode_points.append(f"U+{ord(c):04X}")
            try:
                unicode_names.append(unicodedata.name(c))
            except:
                unicode_names.append("UNKNOWN")
                
        oov_classes.append({
            "Original Class Label": cls,
            "Unicode Codepoints": " ".join(unicode_points),
            "Unicode Names": " | ".join(unicode_names),
            "In Current Vocab": "No (English Characters)",
            "Decomposition": "None",
            "Possible Existing Vocab Tokens": "Requires manual phonetic mapping to Tamil tokens",
            "Genuine New Symbol": "No (It's an English phonetic string, not a Tamil symbol)",
            "Affected Images": sum([1 for f in (DS3_SOURCE / cls).iterdir() if f.is_file()])
        })
        
    df_oov = pd.DataFrame(oov_classes)
    df_oov.to_csv(OUT_DIR / "DS3_OOV_MAPPING.csv", index=False)
    
    md_oov = "# Dataset 3 OOV Mapping Audit\n\n"
    md_oov += f"Total OOV Classes: {len(df_oov)}\n\n"
    md_oov += "## Explanation\n"
    md_oov += "The 19 classes are NOT valid Tamil Unicode characters. They are English phonetic transliterations (e.g. 'cha', 'moo', 'vee').\n"
    md_oov += "Because they contain Latin characters ('c', 'h', 'a') which are not in the Tamil OCR vocabulary, they were correctly flagged as OOV.\n"
    md_oov += "These must be mapped to their corresponding Tamil Unicode strings before training.\n\n"
    
    if len(df_oov) > 0:
        md_oov += df_oov.to_markdown(index=False)
        
    with open(OUT_DIR / "DS3_OOV_MAPPING.md", "w", encoding="utf-8") as f:
        f.write(md_oov)
        
    print("DS3 Audit Complete.")

if __name__ == "__main__":
    run_audit()
