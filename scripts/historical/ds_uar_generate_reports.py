import os
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = str(PROJECT_ROOT / "outputs/dataset_source_audit")
RAW_JSON = os.path.join(OUT_DIR, "raw_audit.json")

def main():
    if not os.path.exists(RAW_JSON):
        print("raw_audit.json not found.")
        return

    with open(RAW_JSON, 'r', encoding='utf-8') as f:
        datasets = json.load(f)

    # 1. Generate Tree Skeleton
    tree_lines = ["DS UAR/"]
    for name in sorted(datasets.keys()):
        info = datasets[name]
        tree_lines.append(f"├── {name}/")
        
        # Add annotation files
        for ann in info["annotation_files"]:
            tree_lines.append(f"│   ├── {os.path.basename(ann)}")
        
        # Add a summary of folders
        folders = set([f.split('\\')[0] for f in info["class_folders"] if '\\' in f] + info["class_folders"])
        top_folders = sorted(list(folders))[:10]
        for tf in top_folders:
            tree_lines.append(f"│   ├── {tf}/")
        if len(folders) > 10:
            tree_lines.append(f"│   ├── ... ({len(folders)} total folders)")
            
        tree_lines.append(f"│   └── <{info['image_files']} image files>")
        
    with open(os.path.join(OUT_DIR, "DS_UAR_COMPLETE_TREE.md"), "w", encoding="utf-8") as f:
        f.write("# DS UAR Complete Tree\n```text\n" + "\n".join(tree_lines) + "\n```")

    # 2. DS_UAR_DATASET_INVENTORY.csv
    inv_rows = []
    comp_rows = []
    
    for name, info in datasets.items():
        if name == "data": continue
        
        g = info.get("geometry", {})
        min_w = g.get("min_w", "N/A")
        max_w = g.get("max_w", "N/A")
        min_h = g.get("min_h", "N/A")
        max_h = g.get("max_h", "N/A")
        channels = "/".join(g.get("channels", [])) if g.get("channels") else "N/A"
        
        exts = ", ".join(f"{k}({v})" for k,v in info["extensions"].items() if k in {'.png','.jpg','.jpeg','.tif','.bmp'})
        
        inv_rows.append({
            "Dataset": name,
            "Path": info["path"],
            "Total Files": info["total_files"],
            "Total Images": info["image_files"],
            "Image Exts": exts,
            "Num Class Folders": len(info["class_folders"]),
            "Annotation Files": len(info["annotation_files"]),
            "Size (bytes)": info["approx_size_bytes"],
            "Min W": min_w,
            "Max W": max_w,
            "Min H": min_h,
            "Max H": max_h,
            "Channels": channels
        })
        
        comp_rows.append({
            "Dataset": name,
            "Source Path": info["path"],
            "Image Count": info["image_files"],
            "Class Count": len(info["class_folders"]),
            "Annotation Type": "CSV/Excel" if info["annotation_files"] else ("Folder Names" if info["class_folders"] else "UNKNOWN"),
            "Label Source": "External" if info["annotation_files"] else "Folder",
            "Image Type": "Unknown",
            "Task Type": "Unknown",
            "Tamil Content": "Unknown",
            "Unicode Status": "REQUIRES MANUAL REVIEW",
            "Current Vocabulary Compatible": "REQUIRES MANUAL REVIEW",
            "OOV Characters": "Unknown",
            "Normalization Required": "Unknown",
            "Conversion Required": "Yes",
            "Train Ready": "No",
            "Character Classifier Candidate": "Yes" if len(info["class_folders"]) > 50 else "Unknown",
            "Sequence OCR Candidate": "Yes" if info["annotation_files"] else "Unknown",
            "Evaluation Candidate": "Unknown",
            "Recommended Destination": "REQUIRES MANUAL REVIEW",
            "Notes": "Auto-generated"
        })

    pd.DataFrame(inv_rows).to_csv(os.path.join(OUT_DIR, "DS_UAR_DATASET_INVENTORY.csv"), index=False)
    pd.DataFrame(comp_rows).to_csv(os.path.join(OUT_DIR, "DS_UAR_COMPATIBILITY_MATRIX.csv"), index=False)
    
    def write_md(filename, title, content):
        with open(os.path.join(OUT_DIR, filename), "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n{content}")

    write_md("DS_UAR_DATASET_INVENTORY.md", "Dataset Inventory", pd.DataFrame(inv_rows).to_markdown(index=False))
    write_md("DS_UAR_COMPATIBILITY_MATRIX.md", "Compatibility Matrix", pd.DataFrame(comp_rows).to_markdown(index=False))
    
    ann_content = ""
    for name, info in datasets.items():
        if name == "data": continue
        ann_content += f"## Dataset {name}\n"
        if not info["annotation_files"]:
            ann_content += "No annotation files found.\n\n"
        for af in info["annotation_files"]:
            ann_content += f"- `{af}`\n"
        ann_content += "\n"
    write_md("DS_UAR_ANNOTATION_AUDIT.md", "Annotation Audit", ann_content)

    write_md("DS_UAR_UNICODE_AUDIT.md", "Unicode Audit", "Requires manual mapping of labels to assess OOV characters.\n")
    write_md("DS_UAR_GEOMETRY_AUDIT.md", "Geometry Audit", pd.DataFrame(inv_rows)[["Dataset", "Min W", "Max W", "Min H", "Max H", "Channels"]].to_markdown(index=False))
    write_md("DS_UAR_DUPLICATE_AUDIT.md", "Duplicate Audit", "Initial scan completed. Full hash overlap requires secondary pass.\n")
    write_md("DS_UAR_RECOMMENDATION.md", "Final Recommendation", "Pending manual review of generated matrices.\n")

if __name__ == "__main__":
    main()
