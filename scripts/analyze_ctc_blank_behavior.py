import os
import sys
import json
import csv
from pathlib import Path
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode
from scripts.dataset_loader import get_dataloader
from models.digitalization.tokenizer import Tokenizer
from utils.device import get_device
from utils.metrics import calculate_cer, calculate_wer

def get_best_checkpoint(checkpoint_dir="checkpoints/recognition/tamil/tamil_full_40epoch"):
    best_path = os.path.join(checkpoint_dir, "best.pth")
    latest_path = os.path.join(checkpoint_dir, "latest.pth")
    if os.path.exists(best_path):
        return best_path
    elif os.path.exists(latest_path):
        return latest_path
    else:
        # Check subdirs if needed, but best is usually root
        return None

def analyze_model():
    device = get_device()
    print(f"Using device: {device}")
    
    working_root = "data/tamil_ocr_dataset"
    vocab_path = os.path.join(working_root, "vocabulary", "tamil_vocab.json")
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    checkpoint_path = get_best_checkpoint()
    if not checkpoint_path:
        print("ERROR: Could not find checkpoint in checkpoints/recognition/tamil")
        return
        
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    val_loader = get_dataloader(
        dataset_root=working_root, 
        language="tamil", 
        split="validation", 
        tokenizer=tokenizer, 
        batch_size=32, 
        num_workers=4,
        max_packets=None,
        target_packet=None,
        replay_previous=False,
        shuffle=False
    )
    
    output_dir = Path("outputs/diagnostics/baseline_ctc")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ----------------------------------------------------
    # METRICS TO TRACK
    # ----------------------------------------------------
    num_samples = 0
    total_cer = 0.0
    total_wer = 0.0
    total_char_acc = 0.0
    total_word_acc = 0.0
    
    gt_char_freq = Counter()
    pred_char_freq = Counter()
    
    confusion_dict = defaultdict(Counter) # true_char -> pred_char counts
    
    # CTC Timestep Analysis
    total_timesteps = 0
    total_blank_timesteps = 0
    empty_decoded_preds = 0
    
    # Sequence Length Analysis
    total_gt_len = 0
    total_pred_len = 0
    
    # Probability and Width stats
    blank_probs = []
    width_ratios = []
    
    # Examples to save
    prediction_examples = []
    
    print("Running diagnostic on validation set...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            if len(batch_data) == 5:
                images, targets, target_lengths, actual_widths, image_paths = batch_data
            else:
                images, targets, target_lengths, actual_widths = batch_data
                image_paths = [f"batch{batch_idx}_img{i}" for i in range(len(images))]
            
            padded_widths = [images.size(3)] * len(images)
                
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images) # [batch, time, classes]
            outputs_perm = outputs.permute(1, 0, 2) # [time, batch, classes]
            outputs_log_probs = outputs_perm.log_softmax(2)
            outputs_probs = torch.exp(outputs_log_probs) # [time, batch, classes]
            
            # Predict
            preds = ctc_decode(outputs) # [batch, time]
            
            offset = 0
            for i in range(len(target_lengths)):
                length = target_lengths[i].item()
                target_seq = targets[offset:offset+length].cpu().tolist()
                offset += length
                
                # Decoder output length calculation (using input lengths for proper valid timesteps)
                valid_time = max(1, actual_widths[i].item() // 4)
                valid_time = min(valid_time, outputs.size(1))
                
                pred_seq_raw = preds[i][:valid_time].cpu().tolist()
                valid_probs = outputs_probs[:valid_time, i, :]
                
                # CTC Decoded sequence
                clean_pred = []
                prev = -1
                for p in pred_seq_raw:
                    if p != 0 and p != prev:
                        clean_pred.append(p)
                    prev = p
                
                ref_text = tokenizer.decode(target_seq)
                hyp_text = tokenizer.decode(clean_pred)
                
                # Basic metrics
                cer = calculate_cer(ref_text, hyp_text)
                wer = calculate_wer(ref_text, hyp_text)
                char_acc = max(0.0, 1.0 - cer)
                word_acc = max(0.0, 1.0 - wer)
                
                total_cer += cer
                total_wer += wer
                total_char_acc += char_acc
                total_word_acc += word_acc
                
                # Timestep Blank Analysis
                blanks_in_sample = sum(1 for p in pred_seq_raw if p == 0)
                total_blank_timesteps += blanks_in_sample
                total_timesteps += valid_time
                
                if len(hyp_text) == 0:
                    empty_decoded_preds += 1
                
                # Probability Analysis
                blank_prob_per_step = valid_probs[:, 0].cpu().numpy()
                avg_blank_prob = float(blank_prob_per_step.mean())
                max_blank_prob = float(blank_prob_per_step.max())
                median_blank_prob = float(torch.median(valid_probs[:, 0]).item())
                avg_non_blank_prob = float(valid_probs[:, 1:].sum(dim=1).mean().item())
                
                blank_probs.append({
                    "avg_blank_prob": avg_blank_prob,
                    "max_blank_prob": max_blank_prob,
                    "median_blank_prob": median_blank_prob,
                    "avg_non_blank_prob": avg_non_blank_prob,
                    "correct": int(wer == 0.0),
                    "gt_len": len(ref_text),
                    "pred_len": len(hyp_text)
                })
                
                # Width Analysis
                actual_w = actual_widths[i].item()
                padded_w = padded_widths[i]
                width_ratios.append({
                    "actual_width": actual_w,
                    "padded_width": padded_w,
                    "ratio": actual_w / padded_w,
                    "blank_percentage": (blanks_in_sample / valid_time) * 100 if valid_time > 0 else 0
                })
                
                # Character frequencies
                for c in ref_text:
                    gt_char_freq[c] += 1
                for c in hyp_text:
                    pred_char_freq[c] += 1
                    
                # Sequence lengths
                total_gt_len += len(ref_text)
                total_pred_len += len(hyp_text)
                
                # Examples
                if len(prediction_examples) < 50:
                    category = "Wrong"
                    if wer == 0:
                        category = "Correct"
                    elif len(hyp_text) == 0:
                        category = "Empty"
                    elif cer < 0.2:
                        category = "Partial"
                        
                    prediction_examples.append({
                        "category": category,
                        "ground_truth": ref_text,
                        "decoded": hyp_text,
                        "raw_argmax": "".join([tokenizer.decode([p]) if p != 0 else '_' for p in pred_seq_raw]),
                        "blank_timesteps": blanks_in_sample,
                        "valid_timesteps": valid_time,
                        "cer": cer
                    })
                    
                num_samples += 1
                
            if num_samples % 500 == 0:
                print(f"Processed {num_samples} samples...")
    
    if num_samples == 0:
        print("No validation samples processed.")
        return
        
    print(f"Total Validation Samples processed: {num_samples}")
    
    # ----------------------------------------------------
    # COMPILE REPORT
    # ----------------------------------------------------
    
    avg_cer = total_cer / num_samples
    avg_wer = total_wer / num_samples
    avg_char_acc = total_char_acc / num_samples
    avg_word_acc = total_word_acc / num_samples
    
    overall_blank_percent = (total_blank_timesteps / total_timesteps) * 100
    empty_prediction_percent = (empty_decoded_preds / num_samples) * 100
    
    unique_gt_chars = len(gt_char_freq)
    unique_pred_chars = len(pred_char_freq)
    
    # Missing chars
    missing_chars = [c for c in gt_char_freq.keys() if c not in pred_char_freq]
    
    # Probability stats
    correct_samples = [x for x in blank_probs if x["correct"] == 1]
    incorrect_samples = [x for x in blank_probs if x["correct"] == 0]
    
    avg_blank_prob_overall = sum(x["avg_blank_prob"] for x in blank_probs) / num_samples
    avg_blank_prob_correct = sum(x["avg_blank_prob"] for x in correct_samples) / len(correct_samples) if correct_samples else 0
    avg_blank_prob_incorrect = sum(x["avg_blank_prob"] for x in incorrect_samples) / len(incorrect_samples) if incorrect_samples else 0
    
    # Width stats
    avg_actual_width = sum(x["actual_width"] for x in width_ratios) / num_samples
    avg_padded_width = sum(x["padded_width"] for x in width_ratios) / num_samples
    avg_width_ratio = sum(x["ratio"] for x in width_ratios) / num_samples
    
    diagnostics_json = {
        "num_samples": num_samples,
        "num_classes_total": tokenizer.num_classes,
        "metrics": {
            "CER": avg_cer,
            "WER": avg_wer,
            "char_accuracy": avg_char_acc,
            "word_accuracy": avg_word_acc
        },
        "character_stats": {
            "gt_unique_characters": unique_gt_chars,
            "pred_unique_characters": unique_pred_chars,
            "num_missing_characters": len(missing_chars),
            "missing_characters_list": missing_chars
        },
        "ctc_blank_stats": {
            "raw_timestep_blank_percentage": overall_blank_percent,
            "empty_decoded_prediction_percentage": empty_prediction_percent,
            "avg_blank_prob_overall": avg_blank_prob_overall,
            "avg_blank_prob_correct_samples": avg_blank_prob_correct,
            "avg_blank_prob_incorrect_samples": avg_blank_prob_incorrect,
        },
        "sequence_lengths": {
            "avg_gt_length": total_gt_len / num_samples,
            "avg_pred_length": total_pred_len / num_samples,
            "ratio_pred_to_gt": total_pred_len / total_gt_len if total_gt_len > 0 else 0
        },
        "width_stats": {
            "avg_actual_width": avg_actual_width,
            "avg_padded_width": avg_padded_width,
            "avg_actual_to_padded_ratio": avg_width_ratio
        }
    }
    
    with open(output_dir / "baseline_ctc_diagnostics.json", "w", encoding='utf-8') as f:
        json.dump(diagnostics_json, f, indent=4, ensure_ascii=False)
        
    with open(output_dir / "baseline_ctc_diagnostics.txt", "w", encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("BASELINE CTC DIAGNOSTICS REPORT\n")
        f.write("="*50 + "\n\n")
        for k, v in diagnostics_json.items():
            f.write(f"{k}:\n")
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    f.write(f"  - {sub_k}: {sub_v}\n")
            else:
                f.write(f"  {v}\n")
            f.write("\n")
            
    # Save examples
    with open(output_dir / "prediction_examples.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["category", "ground_truth", "decoded", "raw_argmax", "blank_timesteps", "valid_timesteps", "cer"])
        writer.writeheader()
        writer.writerows(prediction_examples)
        
    # Save Character Statistics
    with open(output_dir / "character_statistics.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["character", "gt_count", "pred_count", "ratio_pred_to_gt"])
        writer.writeheader()
        all_chars = set(gt_char_freq.keys()) | set(pred_char_freq.keys())
        for c in sorted(list(all_chars)):
            gt_c = gt_char_freq[c]
            pred_c = pred_char_freq[c]
            writer.writerow({
                "character": c,
                "gt_count": gt_c,
                "pred_count": pred_c,
                "ratio_pred_to_gt": (pred_c / gt_c) if gt_c > 0 else float('inf')
            })
            
    print(f"Diagnostics complete. Artifacts saved to {output_dir}")

if __name__ == "__main__":
    analyze_model()
