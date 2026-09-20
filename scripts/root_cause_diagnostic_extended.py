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

# Force UTF-8 encoding for standard output on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.recognition.crnn import CRNN
from scripts.dataset_loader import PacketOCRDataset, crnn_collate_fn
from models.digitalization.tokenizer import Tokenizer
from torch.utils.data import DataLoader, Subset

def run_diagnostics():
    out_dir = Path("outputs/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path("docs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    working_root = "data/tamil_ocr_dataset"
    
    print("Loading tokenizer...")
    vocab_path = os.path.join(working_root, "vocabulary", "tamil_vocab.json")
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Loading dataset...")
    full_dataset = PacketOCRDataset(
        dataset_root=working_root,
        language="tamil",
        split="train",
        tokenizer=tokenizer,
        transform=transform,
        target_packet="packet_001"
    )
    
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    sample_indices = indices[:50]
    
    print("\n--- PHASE 1 & 5: PAIRING & ENCODING ---")
    pairing_results = []
    
    for idx in sample_indices:
        item = full_dataset.data[idx]
        img_path = item["image_path"]
        label = item["label"]
        
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            w, h = 0, 0
            
        target_tokens = tokenizer.encode(label)
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
        
    print("\n--- PHASE 2: VISUAL INSPECTION ---")
    num_cols = 5
    num_rows = 10
    cell_w = 300
    cell_h = 100
    grid_img = Image.new('RGB', (num_cols * cell_w, num_rows * cell_h), color='white')
    draw = ImageDraw.Draw(grid_img)
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
        x, y = c * cell_w, r * cell_h
        try:
            with Image.open(res["image_path"]) as img:
                img_copy = img.copy()
                img_copy.thumbnail((cell_w - 20, cell_h - 40))
                grid_img.paste(img_copy, (x + 10, y + 30))
        except Exception:
            draw.text((x + 10, y + 30), "Error loading image", fill="red")
        label = res["ground_truth"]
        if len(label) > 15: label = label[:12] + "..."
        draw.text((x + 10, y + 5), f"{res['index']}: {label}", fill="black", font=font)
    grid_img.save(out_dir / "tamil_ocr_sample_grid.png")
    
    print("\n--- PHASE 3: PREPROCESSING TRACE ---")
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
        
    print("\n--- PHASE 4: TENSOR DIMENSIONS ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    dummy_input = torch.randn(1, 1, 32, 128).to(device)
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
        
    print("\n--- PHASE 8: DATALOADER BATCHING ---")
    subset = Subset(full_dataset, sample_indices)
    dataloader = DataLoader(subset, batch_size=4, shuffle=False, collate_fn=crnn_collate_fn)
    images, targets, target_lengths = next(iter(dataloader))
    images = images.to(device)
    targets = targets.to(device)
    
    # Check batching integrity
    offset = 0
    batch_integrity = []
    for i in range(len(target_lengths)):
        l = target_lengths[i].item()
        t = targets[offset:offset+l].cpu().tolist()
        offset += l
        dec = tokenizer.decode(t)
        match = (dec == pairing_results[i]["ground_truth"])
        batch_integrity.append({
            "batch_index": i,
            "expected": pairing_results[i]["ground_truth"],
            "actual_decoded": dec,
            "match": match
        })
    with open(out_dir / "dataloader_diagnostic.json", "w", encoding="utf-8") as f:
        json.dump(batch_integrity, f, ensure_ascii=False, indent=4)
        
    print("\n--- PHASE 9: TRAINING CONFIGURATION ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    config_info = {
        "learning_rate": optimizer.param_groups[0]['lr'],
        "optimizer": type(optimizer).__name__,
        "weight_decay": optimizer.param_groups[0]['weight_decay'],
        "model_training_mode": model.training,
        "ctc_reduction": criterion.reduction,
        "ctc_zero_infinity": criterion.zero_infinity,
        "blank_index": criterion.blank
    }
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(config_info, f, indent=4)
        
    print("\n--- PHASE 6 & 7: GRADIENT FLOW & CTC OUTPUTS ---")
    def get_ctc_stats(logits):
        probs = logits.softmax(2)
        blank_probs = probs[:, :, 0].mean().item()
        non_blank_probs = probs[:, :, 1:].max(dim=2)[0].mean().item()
        argmaxes = probs.argmax(dim=2)
        blank_timesteps = (argmaxes == 0).sum().item()
        non_blank_timesteps = (argmaxes != 0).sum().item()
        return {
            "mean_blank_prob": blank_probs,
            "mean_max_non_blank_prob": non_blank_probs,
            "num_blank_timesteps": blank_timesteps,
            "num_non_blank_timesteps": non_blank_timesteps
        }

    # BEFORE TRAINING
    model.train()
    pre_step_weights = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
    outputs_pre = model(images)
    ctc_stats_pre = get_ctc_stats(outputs_pre)
    
    outputs_perm = outputs_pre.permute(1, 0, 2)
    outputs_log_probs = outputs_perm.log_softmax(2)
    input_lengths = torch.full(size=(outputs_perm.size(1),), fill_value=outputs_perm.size(0), dtype=torch.long)
    loss = criterion(outputs_log_probs, targets, input_lengths, target_lengths)
    loss.backward()
    
    grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
    optimizer.step()
    
    # AFTER ONE STEP
    weight_changes = {n: (p - pre_step_weights[n]).norm().item() for n, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        outputs_post = model(images)
    ctc_stats_post = get_ctc_stats(outputs_post)
    
    # AFTER 30 EPOCHS (TINY DATASET)
    try:
        model_30 = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
        ckpt_path = "checkpoints/recognition/tamil/experiments/tiny_overfit_500/epoch_030.pth"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model_30.load_state_dict(ckpt["model_state_dict"])
        model_30.eval()
        with torch.no_grad():
            outputs_30 = model_30(images)
        ctc_stats_30 = get_ctc_stats(outputs_30)
    except Exception as e:
        ctc_stats_30 = {"error": str(e)}

    grad_diag = {
        "loss": loss.item(),
        "grad_norms": grad_norms,
        "weight_changes": weight_changes,
    }
    with open(out_dir / "gradient_diagnostic.json", "w") as f:
        json.dump(grad_diag, f, indent=4)
        
    ctc_diag = {
        "before_training": ctc_stats_pre,
        "after_1_step": ctc_stats_post,
        "after_30_epochs": ctc_stats_30
    }
    with open(out_dir / "ctc_output_diagnostic.json", "w") as f:
        json.dump(ctc_diag, f, indent=4)

    print("Diagnostics complete.")

if __name__ == "__main__":
    run_diagnostics()
