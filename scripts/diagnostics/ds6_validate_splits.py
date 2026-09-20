import os
from pathlib import Path

DS6_SOURCE = (PROJECT_ROOT.parent.parent / "DS UAR/6")

def validate_splits():
    splits = ['train', 'val', 'test']
    
    physical_images = {}
    for split in splits:
        split_dir = DS6_SOURCE / split
        if split_dir.exists():
            for f in os.listdir(split_dir):
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    # store relative path like 'train/1.jpg'
                    rel_path = f"{split}/{f}"
                    physical_images[rel_path] = True
                    
    print(f"Total physical images found: {len(physical_images)}")
    
    txt_files_dir = DS6_SOURCE / "txt_files"
    
    records = []
    
    for split in splits:
        txt_file = txt_files_dir / split / f"{split}_augmented.txt"
        if not txt_file.exists():
            continue
            
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
            
        for line in lines:
            line = line.strip()
            if not line: continue
            
            parts = line.split(maxsplit=1)
            img_path = parts[0].rstrip(',')
            label = parts[1] if len(parts) > 1 else ""
            
            records.append({
                "split": split,
                "image_path": img_path,
                "label": label
            })
            
    print(f"Total annotation records: {len(records)}")
    
    valid_pairs = []
    missing_images = 0
    duplicate_refs = 0
    
    seen_refs = set()
    
    for r in records:
        if r["image_path"] in seen_refs:
            duplicate_refs += 1
        else:
            seen_refs.add(r["image_path"])
            
        if r["image_path"] in physical_images:
            valid_pairs.append(r)
        else:
            missing_images += 1
            
    print(f"Valid image-label pairs: {len(valid_pairs)}")
    print(f"Missing images: {missing_images}")
    print(f"Duplicate references: {duplicate_refs}")
    
    # Check cross-split leakage
    train_refs = set(r["image_path"] for r in records if r["split"] == 'train')
    val_refs = set(r["image_path"] for r in records if r["split"] == 'val')
    test_refs = set(r["image_path"] for r in records if r["split"] == 'test')
    
    leak_train_val = train_refs.intersection(val_refs)
    leak_train_test = train_refs.intersection(test_refs)
    leak_val_test = val_refs.intersection(test_refs)
    
    print(f"Cross-split leakage (train/val): {len(leak_train_val)}")
    print(f"Cross-split leakage (train/test): {len(leak_train_test)}")
    print(f"Cross-split leakage (val/test): {len(leak_val_test)}")

if __name__ == "__main__":
    validate_splits()
