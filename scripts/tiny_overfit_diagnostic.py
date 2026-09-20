import sys
import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from pathlib import Path
import csv
import time
from collections import Counter
import random
import numpy as np

# Force UTF-8 encoding for standard output on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode
from scripts.dataset_loader import PacketOCRDataset, get_dataloader, crnn_collate_fn, AspectRatioPreservingResize
from models.digitalization.tokenizer import Tokenizer
from utils.device import get_device
from utils.metrics import calculate_cer, calculate_wer

def load_config(config_path="config/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_diagnostic():
    print("==================================================")
    print("TINY-DATASET OVERFIT DIAGNOSTIC — NOT A VALID GENERALIZATION RESULT")
    print("==================================================")
    
    config = load_config()
    working_root = config["dataset"]["working_root"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    vocab_path = os.path.join(working_root, "vocabulary", "tamil_vocab.json")
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    full_dataset = PacketOCRDataset(
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
    
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    subset_indices = indices[:500]
    
    diagnostic_dataset = Subset(full_dataset, subset_indices)
    
    dataloader = DataLoader(
        diagnostic_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0,
        collate_fn=crnn_collate_fn
    )
    
    eval_dataloader = DataLoader(
        diagnostic_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=crnn_collate_fn
    )
    
    model = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    optimizer = Adam(model.parameters(), lr=0.0003)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    out_dir = Path("outputs/training/tamil/experiments/tiny_overfit_500")
    ckpt_dir = Path("checkpoints/recognition/tamil/experiments/tiny_overfit_500")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    history_csv = out_dir / "training_history.csv"
    history_json = out_dir / "training_history.json"
    ctc_diag_json = out_dir / "ctc_diagnostics.json"
    
    history_records = []
    ctc_records = {}
    
    epochs = 30
    best_cer = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, (images, targets, target_lengths, actual_widths) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            actual_widths = actual_widths.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)
            outputs_log_probs = outputs.log_softmax(2)
            
            input_lengths = (actual_widths // 4).to(torch.long)
            input_lengths = torch.clamp(input_lengths, max=outputs.size(0))
            
            loss = criterion(outputs_log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        train_loss = epoch_loss / max(1, num_batches)
        duration = time.time() - start_time
        samples_per_sec = len(diagnostic_dataset) / duration
        
        model.eval()
        eval_loss = 0
        eval_batches = 0
        total_cer, total_wer = 0, 0
        
        exact_matches = 0
        all_preds = []
        total_raw_preds = 0
        blank_preds = 0
        
        avg_target_len = 0
        avg_pred_len = 0
        num_samples = 0
        
        fixed_20_results = []
        ctc_5_results = []
        
        with torch.no_grad():
            for batch_idx, (images, targets, target_lengths, actual_widths) in enumerate(eval_dataloader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                actual_widths = actual_widths.to(device, non_blocking=True)
                
                outputs = model(images)
                outputs_perm = outputs.permute(1, 0, 2)
                outputs_log_probs = outputs_perm.log_softmax(2)
                
                input_lengths = (actual_widths // 4).to(torch.long)
                input_lengths = torch.clamp(input_lengths, max=outputs_perm.size(0))
                
                loss = criterion(outputs_log_probs, targets, input_lengths, target_lengths)
                eval_loss += loss.item()
                eval_batches += 1
                
                preds = ctc_decode(outputs)
                total_raw_preds += preds.numel()
                blank_preds += (preds == 0).sum().item()
                
                if batch_idx == 0 and epoch in [1, 5, 10, 20, 30]:
                    for i in range(min(5, images.size(0))):
                        seq_output = outputs[i]
                        seq_log_probs = seq_output.log_softmax(1)
                        seq_probs = seq_output.softmax(1)
                        
                        mean_blank_log_prob = seq_log_probs[:, 0].mean().item()
                        max_non_blank_probs = seq_probs[:, 1:].max(dim=1)[0]
                        mean_max_non_blank = max_non_blank_probs.mean().item()
                        
                        argmaxes = seq_probs.argmax(dim=1)
                        is_blank = (argmaxes == 0)
                        pct_blank_argmax = is_blank.float().mean().item() * 100
                        num_non_blank = (~is_blank).sum().item()
                        
                        ctc_5_results.append({
                            "sample_index": i,
                            "output_tensor_shape": list(seq_output.shape),
                            "target_length": target_lengths[i].item(),
                            "input_sequence_length": seq_output.size(0),
                            "mean_blank_log_prob": mean_blank_log_prob,
                            "mean_max_non_blank_prob": mean_max_non_blank,
                            "percentage_blank_argmax": pct_blank_argmax,
                            "number_non_blank_timesteps": num_non_blank
                        })
                
                offset = 0
                for i in range(len(target_lengths)):
                    length = target_lengths[i].item()
                    target_seq = targets[offset:offset+length].cpu().tolist()
                    offset += length
                    
                    avg_target_len += length
                    
                    try:
                        ref_text = tokenizer.decode(target_seq)
                    except AttributeError:
                        ref_text = "".join([str(c) for c in target_seq])
                        
                    pred_seq = preds[i].cpu().tolist()
                    clean_pred = []
                    prev = -1
                    for p in pred_seq:
                        if p != 0 and p != prev:
                            clean_pred.append(p)
                        prev = p
                        
                    avg_pred_len += len(clean_pred)
                    
                    try:
                        hyp_text = tokenizer.decode(clean_pred)
                    except AttributeError:
                        hyp_text = "".join([str(c) for c in clean_pred])
                        
                    all_preds.append(hyp_text)
                    
                    cer = calculate_cer(ref_text, hyp_text)
                    wer = calculate_wer(ref_text, hyp_text)
                    total_cer += cer
                    total_wer += wer
                    
                    if ref_text == hyp_text:
                        exact_matches += 1
                        
                    if num_samples < 20:
                        fixed_20_results.append({
                            "sample_index": num_samples,
                            "ground_truth": ref_text,
                            "prediction": hyp_text,
                            "cer": cer
                        })
                        
                    num_samples += 1

        eval_loss = eval_loss / max(1, eval_batches)
        avg_cer = total_cer / max(1, num_samples)
        avg_wer = total_wer / max(1, num_samples)
        char_acc = max(0, 1.0 - avg_cer)
        word_acc = max(0, 1.0 - avg_wer)
        
        exact_match_pct = (exact_matches / max(1, num_samples)) * 100
        avg_target_len /= max(1, num_samples)
        avg_pred_len /= max(1, num_samples)
        
        unique_seqs = set(all_preds)
        all_chars = []
        for p in all_preds:
            for c in p:
                all_chars.append(c)
                
        char_counts = Counter(all_chars)
        total_chars = sum(char_counts.values())
        dom_char = ""
        dom_pct = 0.0
        if total_chars > 0:
            dom_char, count = char_counts.most_common(1)[0]
            dom_pct = (count / total_chars) * 100
            
        blank_pct = (blank_preds / total_raw_preds * 100) if total_raw_preds > 0 else 100.0
        diversity_ratio = len(unique_seqs) / max(1, num_samples)
        
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "evaluation_loss": eval_loss,
            "cer": avg_cer,
            "wer": avg_wer,
            "character_accuracy": char_acc,
            "word_accuracy": word_acc,
            "blank_percentage": blank_pct,
            "unique_predicted_characters": len(char_counts),
            "unique_predicted_sequences": len(unique_seqs),
            "average_prediction_length": avg_pred_len,
            "dominant_character": dom_char,
            "dominant_character_percentage": dom_pct,
            "exact_match_count": exact_matches,
            "exact_match_percentage": exact_match_pct,
            "epoch_duration": duration,
            "samples_per_sec": samples_per_sec,
            "average_target_length": avg_target_len,
            "prediction_diversity_ratio": diversity_ratio
        }
        history_records.append(metrics)
        
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f} | CER: {avg_cer:.4f} | Exact: {exact_match_pct:.1f}% | Blanks: {blank_pct:.1f}% | Unique Seqs: {len(unique_seqs)}")
        
        ckpt_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(ckpt_state, ckpt_dir / "latest.pth")
        
        if epoch in [1, 5, 10, 20, 30]:
            torch.save(ckpt_state, ckpt_dir / f"epoch_{epoch:03d}.pth")
            
            with open(out_dir / f"predictions_epoch_{epoch:03d}.json", 'w', encoding='utf-8') as f:
                json.dump(fixed_20_results, f, ensure_ascii=False, indent=4)
                
            ctc_records[f"epoch_{epoch}"] = ctc_5_results
            
        if avg_cer < best_cer:
            best_cer = avg_cer
            torch.save(ckpt_state, ckpt_dir / "best.pth")
            
        with open(history_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writeheader()
            for rec in history_records:
                writer.writerow(rec)
                
        with open(history_json, 'w', encoding='utf-8') as f:
            json.dump(history_records, f, indent=4, ensure_ascii=False)
            
        with open(ctc_diag_json, 'w', encoding='utf-8') as f:
            json.dump(ctc_records, f, indent=4, ensure_ascii=False)
            
    print("\nDiagnostic complete!")

if __name__ == "__main__":
    run_diagnostic()
