import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DS6_SOURCE = (PROJECT_ROOT.parent.parent / "DS UAR/6")
DEST_DIR = (PROJECT_ROOT / "data/train_ready/tamil/sequence_ocr/dataset_06")

for split in ['train', 'val', 'test']:
    src = DS6_SOURCE / split
    dst = DEST_DIR / split
    dst.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "robocopy",
        str(src),
        str(dst),
        "/E",
        "/MT:32",
        "/NP",
        "/NJH",
        "/NJS"
    ]
    
    print(f"Starting copy for {split}...")
    subprocess.run(cmd)
    
print("Robocopy complete.")
