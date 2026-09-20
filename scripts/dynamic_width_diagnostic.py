import os
import sys
import json
import random
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.recognition.crnn import CRNN
from scripts.dataset_loader import PacketOCRDataset, get_dataloader, crnn_collate_fn
from models.digitalization.tokenizer import Tokenizer

def run_diagnostic():
    out_dir = Path("outputs/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    working_root = "data/tamil_ocr_dataset"
    
    vocab_path = os.path.join(working_root, "vocabulary", "tamil_vocab.json")
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    # Old transforms
    old_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # New transform is now inside dataset_loader, we can just instantiate it
    from scripts.dataset_loader import AspectRatioPreservingResize
    new_transform = transforms.Compose([
        AspectRatioPreservingResize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Dummy dataset for raw PIL images
    dataset = PacketOCRDataset(
        dataset_root=working_root,
        language="tamil",
        split="train",
        tokenizer=tokenizer,
        transform=None,
        target_packet="packet_001"
    )
    
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    sample_indices = indices[:500]
    
    print("\n==================================================")
    print("STEP 5 - CTC SAFETY CHECK")
    print("==================================================")
    
    num_satisfy = 0
    num_violate = 0
    max_target_len = 0
    max_req_t = 0
    min_avail_t = 999999
    
    for idx in sample_indices:
        item = dataset.data[idx]
        try:
            img = Image.open(item["image_path"]).convert('L')
            w, h = img.size
        except:
            w, h = 100, 32
            
        target = tokenizer.encode(item["label"])
        t_len = len(target)
        max_target_len = max(max_target_len, t_len)
        
        req_t = 1 # at least 1
        for i in range(1, len(target)):
            if target[i] == target[i-1]:
                req_t += 2 # needs blank in between
            else:
                req_t += 1
                
        max_req_t = max(max_req_t, req_t)
        
        new_w = max(4, round(w * 32 / h))
        avail_t = new_w // 4
        min_avail_t = min(min_avail_t, avail_t)
        
        if avail_t < req_t:
            print(f"VIOLATION: index {idx}, text '{item['label']}', req T={req_t}, avail T={avail_t}")
            num_violate += 1
        else:
            num_satisfy += 1
            
    print(f"Samples satisfying CTC: {num_satisfy}")
    print(f"Samples violating CTC: {num_violate}")
    print(f"Minimum available T: {min_avail_t}")
    print(f"Maximum target length: {max_target_len}")
    print(f"Maximum required T: {max_req_t}")
    
    print("\n==================================================")
    print("STEP 6 - VISUAL VERIFICATION")
    print("==================================================")
    
    num_visuals = 20
    cell_h = 60
    # Left: orig, Mid: old, Right: new
    canvas_w = 300 + 150 + 600
    canvas_h = num_visuals * cell_h
    
    canvas = Image.new('RGB', (canvas_w, canvas_h), color='white')
    draw = ImageDraw.Draw(canvas)
    
    for i in range(num_visuals):
        idx = sample_indices[i]
        item = dataset.data[idx]
        try:
            raw_img = Image.open(item["image_path"]).convert('RGB')
        except:
            continue
            
        # Left: resize raw to fit 300x(cell_h-10) maintaining aspect ratio
        raw_copy = raw_img.copy()
        raw_copy.thumbnail((300, cell_h - 10))
        canvas.paste(raw_copy, (10, i * cell_h + 5))
        
        # Mid: Old transform (32x128)
        old_img = raw_img.copy().resize((128, 32))
        canvas.paste(old_img, (320, i * cell_h + 15))
        
        # Right: New transform (32xProp)
        new_w = max(4, round(raw_img.width * 32 / raw_img.height))
        new_img = raw_img.copy().resize((new_w, 32), Image.Resampling.LANCZOS)
        # paste but clip to 600 width so it doesn't overflow
        crop_w = min(new_w, 580)
        canvas.paste(new_img.crop((0,0,crop_w,32)), (480, i * cell_h + 15))
        
        draw.line([(0, (i+1)*cell_h), (canvas_w, (i+1)*cell_h)], fill="lightgray")
        
    canvas.save(out_dir / "preprocessing_comparison.png")
    print(f"Saved visual comparison to {out_dir / 'preprocessing_comparison.png'}")
    
    print("\n==================================================")
    print("STEP 7 - BATCH DIAGNOSTIC")
    print("==================================================")
    
    dataset.transform = new_transform
    subset = Subset(dataset, sample_indices)
    dataloader = DataLoader(subset, batch_size=4, shuffle=False, collate_fn=crnn_collate_fn)
    
    images, targets, target_lengths, actual_widths = next(iter(dataloader))
    
    device = torch.device('cpu')
    model = CRNN(tokenizer.num_classes, feature_size=2048)
    
    outputs = model.encoder(images)
    b, c, h, w_cnn = outputs.size()
    
    print("Input tensor shape:", images.shape)
    print("Actual image widths:", actual_widths.tolist())
    print("CNN output shape:", outputs.shape)
    print("Temporal length T (padded):", w_cnn)
    print("Target lengths:", target_lengths.tolist())
    print("Calculated input_lengths (for CTC):", (actual_widths // 4).tolist())
    
if __name__ == "__main__":
    run_diagnostic()
