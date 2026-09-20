import os
import sys
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode
from scripts.dataset_loader import PacketOCRDataset, get_dataloader, crnn_collate_fn, AspectRatioPreservingResize
from models.digitalization.tokenizer import Tokenizer
from utils.metrics import calculate_cer, calculate_wer

def calculate_l2_and_cosine(features):
    # features shape: [4, D]
    b = features.size(0)
    l2_dists = []
    cos_sims = []
    for i in range(b):
        for j in range(i+1, b):
            l2 = F.pairwise_distance(features[i:i+1], features[j:j+1]).item()
            cos = F.cosine_similarity(features[i:i+1], features[j:j+1]).item()
            l2_dists.append(l2)
            cos_sims.append(cos)
    return {
        "l2_mean": float(np.mean(l2_dists)) if l2_dists else 0,
        "l2_std": float(np.std(l2_dists)) if l2_dists else 0,
        "cos_mean": float(np.mean(cos_sims)) if cos_sims else 0,
        "cos_std": float(np.std(cos_sims)) if cos_sims else 0
    }

def get_parameter_stats(model):
    stats = {}
    for name, param in model.named_parameters():
        stats[name] = {
            "mean": float(param.data.mean().item()),
            "std": float(param.data.std().item()),
            "min": float(param.data.min().item()),
            "max": float(param.data.max().item())
        }
    return stats

def get_gradient_stats(model):
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            stats[name] = {
                "grad_norm": float(param.grad.norm().item()),
                "grad_min": float(param.grad.min().item()),
                "grad_max": float(param.grad.max().item())
            }
        else:
            stats[name] = {"grad_norm": 0, "grad_min": 0, "grad_max": 0}
    return stats

def run_diagnostic():
    out_dir = Path("outputs/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    working_root = "data/tamil_ocr_dataset"
    
    vocab_path = os.path.join(working_root, "vocabulary", "tamil_vocab.json")
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    dataset = PacketOCRDataset(
        dataset_root=working_root,
        language="tamil",
        split="train",
        tokenizer=tokenizer,
        transform=transforms.Compose([
            AspectRatioPreservingResize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        target_packet="packet_001"
    )
    
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Pick 4 random samples
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    sample_indices = indices[:4]
    
    subset = Subset(dataset, sample_indices)
    dataloader = DataLoader(subset, batch_size=4, shuffle=False, collate_fn=crnn_collate_fn)
    
    images, targets, target_lengths, actual_widths = next(iter(dataloader))
    
    print("==================================================")
    print("TEST 1 — SINGLE-BATCH MEMORIZATION INFO")
    print("==================================================")
    
    offset = 0
    for i in range(4):
        idx = sample_indices[i]
        item = dataset.data[idx]
        print(f"Sample {i}:")
        print(f"  Image path: {item['image_path']}")
        print(f"  Ground-truth text: {item['label']}")
        from PIL import Image
        orig_img = Image.open(item["image_path"])
        print(f"  Original dimensions: {orig_img.size}")
        
        t_len = target_lengths[i].item()
        t_seq = targets[offset:offset+t_len].cpu().tolist()
        offset += t_len
        print(f"  Target length: {t_len}")
        print(f"  Target seq: {t_seq}")
        
    print(f"Padded batch dimensions: {images.shape}")
    print(f"Actual image widths: {actual_widths.tolist()}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    
    images = images.to(device)
    targets = targets.to(device)
    actual_widths = actual_widths.to(device)
    
    print("\n==================================================")
    print("TEST 9 — INITIALIZATION AUDIT")
    print("==================================================")
    init_stats = get_parameter_stats(model)
    # Identify layers dynamically
    param_names = list(init_stats.keys())
    sample_layers = []
    if len(param_names) > 0: sample_layers.append(param_names[0])
    for name in param_names:
        if ('rnn' in name or 'lstm' in name) and 'weight' in name:
            sample_layers.append(name)
            break
    for name in reversed(param_names):
        if 'weight' in name:
            sample_layers.append(name)
            break
            
    # Just print a few representative layers
    for layer in sample_layers:
        print(f"{layer}: {init_stats.get(layer, 'Not Found')}")
        
    print("\n==================================================")
    print("TEST 4 — VARIABLE WIDTH / MASKING CHECK")
    print("==================================================")
    
    model.eval()
    with torch.no_grad():
        cnn_out = model.encoder(images)
        b, c, h, w_cnn = cnn_out.size()
        
        # Test 7: CNN Feature Diversity
        # pool over h and w_cnn to get [b, c]
        cnn_pooled = cnn_out.mean(dim=(2, 3))
        cnn_div = calculate_l2_and_cosine(cnn_pooled)
        with open(out_dir / "cnn_feature_diversity.json", "w") as f:
            json.dump(cnn_div, f, indent=2)
            
        # Test 8: LSTM Feature Diversity
        conv_out = cnn_out.permute(0, 3, 1, 2).reshape(b, w_cnn, c * h)
        rnn_out = model.sequence(conv_out) # [b, w, hidden]
        # pool over w
        rnn_pooled = rnn_out.mean(dim=1)
        lstm_div = calculate_l2_and_cosine(rnn_pooled)
        with open(out_dir / "lstm_feature_diversity.json", "w") as f:
            json.dump(lstm_div, f, indent=2)
            
        print("CNN output dimensions:", cnn_out.shape)
        print("CNN temporal length T:", w_cnn)
        input_lengths = (actual_widths // 4).to(torch.long)
        input_lengths = torch.clamp(input_lengths, max=cnn_out.size(3))
        print("CTC input_lengths calculated:", input_lengths.tolist())
        print("Note: The CTCLoss receives exactly these input_lengths, which successfully masks out the padded regions.")
    
    print("\n==================================================")
    print("TEST 10 — CTC LOGIT AUDIT (Step 0)")
    print("==================================================")
    with torch.no_grad():
        outputs = model(images)
        outputs_perm = outputs.permute(1, 0, 2)
        outputs_log_probs = outputs_perm.log_softmax(2)
        seq_probs = outputs_perm.softmax(2)
        
        print("Logits (Step 0):")
        print(f"  Mean: {outputs_perm.mean().item():.4f}")
        print(f"  Std: {outputs_perm.std().item():.4f}")
        print(f"  Blank Probability: {seq_probs[:,:,0].mean().item() * 100:.2f}%")
        
        # Top-5 classes
        top5 = seq_probs.mean(dim=(0,1)).topk(5)
        print(f"  Top-5 classes: {top5.indices.tolist()} with probs {top5.values.tolist()}")
        
    print("\n==================================================")
    print("TEST 5 — OVERFIT WITH CURRENT ARCHITECTURE")
    print("==================================================")
    
    model.train()
    
    saved_params_before = None
    
    history = []
    
    weight_deltas = None
    grad_stats = None
    
    for epoch in range(1, 201):
        optimizer.zero_grad()
        
        if epoch == 1:
            saved_params_before = {name: param.clone().detach() for name, param in model.named_parameters()}
            
        outputs = model(images)
        outputs_perm = outputs.permute(1, 0, 2)
        outputs_log_probs = outputs_perm.log_softmax(2)
        
        input_lengths = (actual_widths // 4).to(torch.long)
        input_lengths = torch.clamp(input_lengths, max=outputs_perm.size(0))
        
        loss = criterion(outputs_log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        
        if epoch == 200:
            grad_stats = get_gradient_stats(model)
            
        optimizer.step()
        
        if epoch == 1:
            weight_deltas = {}
            for name, param in model.named_parameters():
                delta = torch.abs(param.data - saved_params_before[name])
                weight_deltas[name] = {
                    "mean_delta": float(delta.mean().item()),
                    "max_delta": float(delta.max().item())
                }
                
        # Calculate metrics
        with torch.no_grad():
            preds = ctc_decode(outputs)
            
            total_cer, exact_matches, total_len = 0, 0, 0
            blank_count = (preds == 0).sum().item()
            total_elements = preds.numel()
            blank_pct = (blank_count / total_elements) * 100
            
            unique_seqs = set()
            
            offset = 0
            for i in range(4):
                t_len = target_lengths[i].item()
                t_seq = targets[offset:offset+t_len].cpu().tolist()
                offset += t_len
                
                try:
                    ref_text = tokenizer.decode(t_seq)
                except:
                    ref_text = str(t_seq)
                    
                p_seq = preds[i].cpu().tolist()
                clean_p = []
                prev = -1
                for p in p_seq:
                    if p != 0 and p != prev:
                        clean_p.append(p)
                    prev = p
                    
                try:
                    pred_text = tokenizer.decode(clean_p)
                except:
                    pred_text = str(clean_p)
                    
                unique_seqs.add(pred_text)
                
                cer = calculate_cer(ref_text, pred_text)
                total_cer += cer
                if cer == 0:
                    exact_matches += 1
                total_len += len(clean_p)
                
            avg_cer = total_cer / 4
            exact_match_pct = (exact_matches / 4) * 100
            avg_pred_len = total_len / 4
            
        history.append({
            "epoch": epoch,
            "loss": float(loss.item()),
            "cer": float(avg_cer),
            "exact_match": float(exact_match_pct),
            "blank_pct": float(blank_pct),
            "unique_seqs": len(unique_seqs),
            "avg_pred_len": float(avg_pred_len)
        })
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | CER: {avg_cer:.4f} | Exact: {exact_match_pct:.1f}% | Blanks: {blank_pct:.1f}% | Unique: {len(unique_seqs)}")
            
        if epoch == 1 or epoch == 10:
            print(f"\n--- TEST 10 - CTC LOGIT AUDIT (Step {epoch}) ---")
            seq_probs = outputs_perm.softmax(2)
            print(f"  Mean Logits: {outputs_perm.mean().item():.4f}")
            print(f"  Blank Probability: {seq_probs[:,:,0].mean().item() * 100:.2f}%")
            top5 = seq_probs.mean(dim=(0,1)).topk(5)
            print(f"  Top-5 classes: {top5.indices.tolist()} with probs {top5.values.tolist()}\n")
            
    print("\n==================================================")
    print("TEST 3 — WEIGHT UPDATE VERIFICATION (Epoch 1 Deltas)")
    print("==================================================")
    for layer in sample_layers:
        print(f"{layer} deltas: {weight_deltas.get(layer, 'Not Found')}")
        
    print("\n==================================================")
    print("TEST 2 — GRADIENT DIAGNOSTIC (Epoch 200)")
    print("==================================================")
    for layer in sample_layers:
        print(f"{layer} grads: {grad_stats.get(layer, 'Not Found')}")
        
    # Save combined report
    combined_report = {
        "cnn_diversity": cnn_div,
        "lstm_diversity": lstm_div,
        "initialization_sample": {k: init_stats[k] for k in sample_layers if k in init_stats},
        "weight_deltas_sample": {k: weight_deltas[k] for k in sample_layers if k in weight_deltas},
        "final_gradients_sample": {k: grad_stats[k] for k in sample_layers if k in grad_stats},
        "history": history
    }
    
    with open(out_dir / "crnn_architecture_diagnostic.json", "w") as f:
        json.dump(combined_report, f, indent=2)
        
    print("\nDiagnostic finished. Results saved to outputs/diagnostics/")

if __name__ == "__main__":
    run_diagnostic()
