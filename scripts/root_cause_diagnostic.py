import os
import sys
import json
import random
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt

# Force UTF-8 encoding for standard output on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.recognition.crnn import CRNN
from scripts.dataset_loader import PacketOCRDataset, crnn_collate_fn
from models.digitalization.tokenizer import Tokenizer

def run_diagnostics():
    out_dir = Path("outputs/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path("docs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    working_root = "data/tamil_ocr_dataset"
    
    print("Loading tokenizer...")
    vocab_path = os.path.join(working_root, "vocabulary", "tamil_vocab.json")
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    print("Loading dataset...")
    # Get raw dataset first for pairing verification
    raw_dataset = PacketOCRDataset(
        dataset_root=working_root,
        language="tamil",
        split="train",
        tokenizer=tokenizer,
        transform=None,
        target_packet="packet_001"
    )
    
    # Deterministic 50 samples
    seed = 42
    random.seed(seed)
    indices = list(range(len(raw_dataset)))
    random.shuffle(indices)
    sample_indices = indices[:50]
    
    print("1. & 4. Verifying Image/Label Pairing and Target Encoding...")
    pairing_results = []
    
    for idx in sample_indices:
        item = raw_dataset.data[idx]
        img_path = item["image_path"]
        label = item["label"]
        
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            w, h = 0, 0
            
        target_tokens = tokenizer.encode(label)
        
        # Verify decoding
        try:
            decoded_text = tokenizer.decode(target_tokens)
        except AttributeError:
            decoded_text = "".join([str(c) for c in target_tokens])
            
        match = (label == decoded_text)
        
        pairing_results.append({
            "index": idx,
            "image_path": img_path,
            "ground_truth": label,
            "image_width": w,
            "image_height": h,
            "target_character_length": len(label),
            "target_token_ids": target_tokens,
            "decoded_text": decoded_text,
            "encoding_match": match
        })
        
    with open(out_dir / "dataset_pairing_samples.json", "w", encoding="utf-8") as f:
        json.dump(pairing_results, f, ensure_ascii=False, indent=4)
        
    print("2. Verifying Image Preprocessing...")
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    preprocessed_info = {}
    sample_img_path = pairing_results[0]["image_path"]
    with Image.open(sample_img_path).convert('L') as raw_img:
        orig_w, orig_h = raw_img.size
        tensor_img = transform(raw_img)
        
        preprocessed_info = {
            "original_dimensions": [orig_w, orig_h],
            "processed_dimensions": [128, 32],
            "tensor_shape": list(tensor_img.shape),
            "dtype": str(tensor_img.dtype),
            "min_pixel": tensor_img.min().item(),
            "max_pixel": tensor_img.max().item(),
            "mean": tensor_img.mean().item(),
            "std": tensor_img.std().item()
        }
    with open(out_dir / "preprocessing_diagnostic.json", "w") as f:
        json.dump(preprocessed_info, f, indent=4)
        
    print("3. Verifying Temporal Dimension...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    
    dummy_input = torch.randn(1, 1, 32, 128).to(device)
    
    temporal_info = {}
    with torch.no_grad():
        cnn_out = model.encoder(dummy_input)
        b, c, h, w = cnn_out.size()
        
        rnn_in = cnn_out.permute(0, 3, 1, 2).reshape(b, w, c * h)
        
        rnn_out = model.sequence(rnn_in)
        final_out = model.head(rnn_out)
        
        temporal_info = {
            "input_shape": list(dummy_input.shape),
            "cnn_output_shape": [b, c, h, w],
            "sequence_length_T": w,
            "feature_size": c * h,
            "lstm_input_size": c * h,
            "lstm_output_shape": list(rnn_out.shape),
            "ctc_output_shape": list(final_out.shape)
        }
    with open(out_dir / "temporal_diagnostic.json", "w") as f:
        json.dump(temporal_info, f, indent=4)

    print("5. & 6. Verifying Gradient Flow & Model Output...")
    from torch.utils.data import Subset, DataLoader
    subset = Subset(PacketOCRDataset(working_root, "tamil", "train", tokenizer, transform=transform, target_packet="packet_001"), sample_indices)
    dataloader = DataLoader(subset, batch_size=4, shuffle=False, collate_fn=crnn_collate_fn)
    
    images, targets, target_lengths = next(iter(dataloader))
    images = images.to(device)
    targets = targets.to(device)
    
    model.train()
    
    # Store pre-step weights
    pre_step_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            pre_step_weights[name] = param.clone().detach()
            
    # Pre-step outputs
    outputs_pre = model(images)
    probs_pre = outputs_pre.softmax(2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    optimizer.zero_grad()
    outputs = model(images)
    outputs_perm = outputs.permute(1, 0, 2)
    outputs_log_probs = outputs_perm.log_softmax(2)
    input_lengths = torch.full(size=(outputs_perm.size(1),), fill_value=outputs_perm.size(0), dtype=torch.long)
    
    loss = criterion(outputs_log_probs, targets, input_lengths, target_lengths)
    loss.backward()
    
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
            
    optimizer.step()
    
    # Post-step comparison
    weight_changes = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            diff = (param - pre_step_weights[name]).norm().item()
            weight_changes[name] = diff
            
    # Post-step outputs
    with torch.no_grad():
        outputs_post = model(images)
        probs_post = outputs_post.softmax(2)
        
    gradient_diagnostic = {
        "loss": loss.item(),
        "grad_norms": grad_norms,
        "weight_changes": weight_changes,
        "pre_step_probs_mean_blank": probs_pre[:, :, 0].mean().item(),
        "post_step_probs_mean_blank": probs_post[:, :, 0].mean().item()
    }
    with open(out_dir / "gradient_diagnostic.json", "w") as f:
        json.dump(gradient_diagnostic, f, indent=4)
        
    print("7. Creating Visual Data Diagnostic...")
    # Create contact sheet
    num_cols = 5
    num_rows = 10
    cell_w = 300
    cell_h = 100
    
    grid_img = Image.new('RGB', (num_cols * cell_w, num_rows * cell_h), color='white')
    draw = ImageDraw.Draw(grid_img)
    
    # Try to load Nirmala UI for Tamil support, fallback to default
    try:
        font = ImageFont.truetype("Nirmala.ttf", 20)
    except IOError:
        try:
            font = ImageFont.truetype("latha.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
            
    for i, res in enumerate(pairing_results):
        r = i // num_cols
        c = i % num_cols
        
        x = c * cell_w
        y = r * cell_h
        
        try:
            with Image.open(res["image_path"]) as img:
                img_copy = img.copy()
                img_copy.thumbnail((cell_w - 20, cell_h - 40))
                grid_img.paste(img_copy, (x + 10, y + 30))
        except Exception as e:
            draw.text((x + 10, y + 30), f"Error loading image", fill="red")
            
        label = res["ground_truth"]
        # Limit label length for display
        if len(label) > 15:
            label = label[:12] + "..."
        draw.text((x + 10, y + 5), f"{res['index']}: {label}", fill="black", font=font)
        
    grid_img.save(out_dir / "tamil_ocr_sample_grid.png")
    
    print("Diagnostics complete. Files saved to outputs/diagnostics/")

if __name__ == "__main__":
    run_diagnostics()
