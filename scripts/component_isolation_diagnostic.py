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
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode
from scripts.dataset_loader import PacketOCRDataset, crnn_collate_fn, AspectRatioPreservingResize
from models.digitalization.tokenizer import Tokenizer
from utils.metrics import calculate_cer

def calculate_l2_and_cosine(features):
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

class SyntheticBarcodeDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, tokenizer):
        self.samples = []
        self.tokenizer = tokenizer
        vocab = [v for v in tokenizer.char_map.keys() if v != '<blank>' and v != '<UNK>']
        for _ in range(num_samples):
            chars = [random.choice(vocab) for _ in range(random.randint(1, 3))]
            label = "".join(chars)
            
            img = Image.new('L', (120, 32), color=255)
            d = ImageDraw.Draw(img)
            for idx, c in enumerate(chars):
                char_idx = tokenizer.char_map[c]
                h = max(2, (char_idx % 30))
                d.rectangle([idx*30+5, 32-h, idx*30+20, 32], fill=0)
                
            self.samples.append((img, label))
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        img_tensor = self.transform(img)
        target = self.tokenizer.encode(label)
        return img_tensor, torch.tensor(target, dtype=torch.long), img.width

def train_overfit(model, dataloader, epochs, name, device, criterion, tokenizer):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    model.train()
    print(f"\n--- TEST: {name} (Max {epochs} Epochs) ---")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        try:
            images, targets, target_lengths, actual_widths = next(iter(dataloader))
        except StopIteration:
            break
            
        images, targets, actual_widths = images.to(device), targets.to(device), actual_widths.to(device)
        
        outputs = model(images)
        outputs_perm = outputs.permute(1, 0, 2)
        outputs_log_probs = outputs_perm.log_softmax(2)
        
        input_lengths = (actual_widths // 4).to(torch.long)
        input_lengths = torch.clamp(input_lengths, max=outputs_perm.size(0))
        
        loss = criterion(outputs_log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                preds = ctc_decode(outputs)
                cer = 0
                offset = 0
                for i in range(images.size(0)):
                    t_len = target_lengths[i].item()
                    t_seq = targets[offset:offset+t_len].cpu().tolist()
                    offset += t_len
                    try: ref_text = tokenizer.decode(t_seq)
                    except: ref_text = str(t_seq)
                    
                    p_seq = preds[i].cpu().tolist()
                    clean_p = [p for i, p in enumerate(p_seq) if p != 0 and (i == 0 or p != p_seq[i-1])]
                    try: pred_text = tokenizer.decode(clean_p)
                    except: pred_text = str(clean_p)
                    
                    cer += calculate_cer(ref_text, pred_text)
                cer /= images.size(0)
                
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | CER: {cer:.4f}")
            if cer == 0.0:
                print(f"SUCCESS: {name} memorized perfectly at epoch {epoch}!")
                return True
    
    print(f"FAILED: {name} failed to memorize (CER > 0).")
    return False

def run_diagnostic():
    out_dir = Path("outputs/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_path = "data/tamil_ocr_dataset/vocabulary/tamil_vocab.json"
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    dataset = PacketOCRDataset(
        dataset_root="data/tamil_ocr_dataset", language="tamil", split="train",
        tokenizer=tokenizer, target_packet="packet_001",
        transform=transforms.Compose([
            AspectRatioPreservingResize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ])
    )
    
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Select 4 samples
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    sample_indices = indices[:4]
    
    # Find one very short sample
    short_idx = None
    for i in range(1000):
        if len(dataset[i][1]) <= 3:
            short_idx = i
            break
    if short_idx is None: short_idx = 0
    
    loader_4 = DataLoader(Subset(dataset, sample_indices), batch_size=4, collate_fn=crnn_collate_fn)
    loader_1 = DataLoader(Subset(dataset, [sample_indices[0]]), batch_size=1, collate_fn=crnn_collate_fn)
    loader_short = DataLoader(Subset(dataset, [short_idx]), batch_size=1, collate_fn=crnn_collate_fn)
    
    synth_dataset = SyntheticBarcodeDataset(4, tokenizer)
    loader_synth = DataLoader(synth_dataset, batch_size=4, collate_fn=crnn_collate_fn)
    
    model = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    
    images, targets, target_lengths, actual_widths = next(iter(loader_4))
    images, targets, actual_widths = images.to(device), targets.to(device), actual_widths.to(device)
    
    print("\n==================================================")
    print("TEST A — VERIFY ACTUAL CTC INPUT LENGTHS")
    print("==================================================")
    model.eval()
    with torch.no_grad():
        cnn_out = model.encoder(images)
        w_cnn = cnn_out.size(3)
        input_lengths = (actual_widths // 4).to(torch.long)
        input_lengths = torch.clamp(input_lengths, max=w_cnn)
        for i in range(4):
            print(f"Sample {i}: Actual Width={actual_widths[i].item()} -> T={w_cnn} -> CTC Valid Length={input_lengths[i].item()} (Target={target_lengths[i].item()})")
            
    print("\n==================================================")
    print("TEST B — CHECK WHETHER PADDING CONTAMINATES FEATURES")
    print("==================================================")
    # Run sample 0 unpadded
    img0_unpadded, _, _, width0 = next(iter(loader_1))
    img0_unpadded = img0_unpadded.to(device)
    with torch.no_grad():
        cnn0_unpadded = model.encoder(img0_unpadded)
        
        # Valid length
        valid_t = input_lengths[0].item()
        
        cnn0_padded = cnn_out[0:1] # [1, C, H, W_cnn]
        
        feat_unpadded = cnn0_unpadded[:, :, :, :valid_t]
        feat_padded = cnn0_padded[:, :, :, :valid_t]
        
        diff = torch.abs(feat_unpadded - feat_padded)
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()
        
        cos_sim = F.cosine_similarity(feat_unpadded.flatten(1), feat_padded.flatten(1)).item()
        print(f"Mean Abs Diff: {mean_diff:.6f}, Max Abs Diff: {max_diff:.6f}, Cosine Sim: {cos_sim:.6f}")
        if cos_sim < 0.99: print("WARNING: Padding significantly contaminates valid features!")
        else: print("PASS: Padding does not corrupt valid features.")
        
    print("\n==================================================")
    print("TEST C, D, E — FEATURE & HEAD DIVERSITY")
    print("==================================================")
    with torch.no_grad():
        b, c, h, w_cnn = cnn_out.size()
        cnn_pooled = cnn_out.mean(dim=(2, 3))
        cnn_div = calculate_l2_and_cosine(cnn_pooled)
        with open(out_dir / "cnn_feature_diversity.json", "w") as f: json.dump(cnn_div, f, indent=2)
        print(f"CNN Cosine Similarity Mean: {cnn_div['cos_mean']:.4f}")
        
        conv_out = cnn_out.permute(0, 3, 1, 2).reshape(b, w_cnn, c * h)
        rnn_out = model.sequence(conv_out)
        rnn_pooled = rnn_out.mean(dim=1)
        lstm_div = calculate_l2_and_cosine(rnn_pooled)
        with open(out_dir / "lstm_feature_diversity.json", "w") as f: json.dump(lstm_div, f, indent=2)
        print(f"LSTM Cosine Similarity Mean: {lstm_div['cos_mean']:.4f}")
        
        logits = model.head(rnn_out)
        seq_probs = logits.softmax(2)
        entropy = -(seq_probs * seq_probs.clamp(min=1e-9).log()).sum(-1).mean().item()
        print(f"CTC Logits Mean: {logits.mean().item():.4f}, Std: {logits.std().item():.4f}")
        print(f"CTC Blank Probability: {seq_probs[:,:,0].mean().item()*100:.2f}%")
        print(f"CTC Entropy: {entropy:.4f}")

    print("\n==================================================")
    print("TEST F, G — GRADIENT ATTRIBUTION & PARAMETER UPDATE")
    print("==================================================")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    
    saved_params = {name: p.clone().detach() for name, p in model.named_parameters()}
    
    outputs = model(images)
    outputs_perm = outputs.permute(1, 0, 2)
    loss = criterion(outputs_perm.log_softmax(2), targets, input_lengths, target_lengths)
    loss.backward()
    
    grad_stats = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            norm = p.grad.norm().item()
            status = "HEALTHY"
            if norm < 1e-5: status = "VANISHING"
            if norm > 10.0: status = "EXPLODING"
            if norm == 0.0: status = "DEAD"
            grad_stats[name] = {"norm": norm, "status": status}
            
    optimizer.step()
    
    update_stats = {}
    for name, p in model.named_parameters():
        delta = torch.abs(p.data - saved_params[name])
        update_stats[name] = {"mean_change": delta.mean().item()}
        
    for name in ['encoder.features.0.weight', 'sequence.lstm.weight_ih_l0', 'head.linear.weight']:
        if name in grad_stats:
            print(f"{name} Gradients: {grad_stats[name]['norm']:.2e} ({grad_stats[name]['status']})")
            print(f"{name} Updates: {update_stats[name]['mean_change']:.2e}")
            
    print("\n==================================================")
    print("TEST H, I, J — ISOLATED OVERFITTING")
    print("==================================================")
    
    # Reinitialize model for overfit tests
    model_h = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    res_h = train_overfit(model_h, loader_1, 500, "TEST H (Single Image)", device, criterion, tokenizer)
    
    model_i = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    res_i = train_overfit(model_i, loader_short, 500, "TEST I (Short Label)", device, criterion, tokenizer)
    
    model_j = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    res_j = train_overfit(model_j, loader_synth, 500, "TEST J (Synthetic Dataset)", device, criterion, tokenizer)
    
    # Save combined report
    combined_report = {
        "cnn_diversity": cnn_div,
        "lstm_diversity": lstm_div,
        "grad_stats": {k: grad_stats[k] for k in ['encoder.features.0.weight', 'sequence.lstm.weight_ih_l0', 'head.linear.weight'] if k in grad_stats},
        "overfit_results": {
            "single_image": res_h,
            "short_label": res_i,
            "synthetic_control": res_j
        }
    }
    with open(out_dir / "crnn_component_isolation.json", "w") as f:
        json.dump(combined_report, f, indent=2)

if __name__ == "__main__":
    run_diagnostic()
