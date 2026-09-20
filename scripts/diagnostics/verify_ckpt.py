import os
import hashlib
import torch
import datetime

ckpt_path = "checkpoints/recognition/tamil/ocr_balanced_temporal/best.pth"
st = os.stat(ckpt_path)
print("File Size:", st.st_size, "bytes")
print("Modified:", datetime.datetime.fromtimestamp(st.st_mtime))

with open(ckpt_path, "rb") as f:
    hasher = hashlib.sha256()
    for chunk in iter(lambda: f.read(4096), b""):
        hasher.update(chunk)
print("SHA-256:", hasher.hexdigest())

checkpoint = torch.load(ckpt_path, map_location="cpu")
print("Keys:", list(checkpoint.keys()))
if "epoch" in checkpoint:
    print("Epoch:", checkpoint["epoch"])
if "model_config" in checkpoint:
    print("Config present in checkpoint.")
else:
    print("No config metadata in checkpoint.")
