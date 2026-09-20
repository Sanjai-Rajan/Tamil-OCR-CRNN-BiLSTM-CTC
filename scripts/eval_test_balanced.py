import sys
import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
from pathlib import Path
import time
from collections import Counter
import editdistance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode
from scripts.dataset_loader import get_dataloader
from models.digitalization.tokenizer import Tokenizer

def calculate_cer(ref, hyp):
    ref = ref.strip()
    hyp = hyp.strip()
    if len(ref) == 0: return float(len(hyp))
    return editdistance.eval(ref, hyp) / len(ref)

def calculate_wer(ref, hyp):
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if len(ref_words) == 0: return float(len(hyp_words))
    return editdistance.eval(ref_words, hyp_words) / len(ref_words)

def get_bucket(target_len):
    if target_len <= 5: return "1-5"
    if target_len <= 10: return "6-10"
    if target_len <= 15: return "11-15"
    return "16+"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoints/recognition/tamil/ocr_balanced_temporal/best.pth"
    working_root = "data/tamil_ocr_dataset"
    vocab_path = os.path.join(working_root, "vocabulary", "tamil_vocab.json")
    
    tokenizer = Tokenizer(vocab_path=vocab_path)
    model = CRNN(tokenizer.num_classes, feature_size=2048, less_downsample=True).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataloader = get_dataloader(
        dataset_root=working_root,
        language="tamil",
        split="test",
        tokenizer=tokenizer,
        batch_size=128,
        num_workers=4,
        max_packets=None,
        target_packet=None,
        replay_previous=False,
        shuffle=False
    )
    
    val_loss = 0.0
    total_cer, total_wer = 0.0, 0.0
    num_samples, total_raw_preds, blank_preds, exact_matches = 0, 0, 0, 0
    
    buckets = {
        "1-5": {"count": 0, "target_len_sum": 0, "pred_len_sum": 0, "cer_sum": 0, "exact_matches": 0},
        "6-10": {"count": 0, "target_len_sum": 0, "pred_len_sum": 0, "cer_sum": 0, "exact_matches": 0},
        "11-15": {"count": 0, "target_len_sum": 0, "pred_len_sum": 0, "cer_sum": 0, "exact_matches": 0},
        "16+": {"count": 0, "target_len_sum": 0, "pred_len_sum": 0, "cer_sum": 0, "exact_matches": 0},
    }
    
    target_length_sum = 0
    pred_length_sum = 0
    
    start_time = time.time()
    with torch.no_grad():
        for batch_data in dataloader:
            if len(batch_data) == 5:
                images, targets, target_lengths, actual_widths, _ = batch_data
            else:
                images, targets, target_lengths, actual_widths = batch_data
            
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            preds = ctc_decode(outputs)
            
            total_raw_preds += preds.numel()
            blank_preds += (preds == 0).sum().item()
            
            offset = 0
            for i in range(len(target_lengths)):
                length = target_lengths[i].item()
                target_seq = targets[offset:offset+length].cpu().tolist()
                offset += length
                
                try: ref_text = tokenizer.decode(target_seq)
                except: ref_text = "".join([str(c) for c in target_seq])
                    
                pred_seq = preds[i].cpu().tolist()
                clean_pred = []
                prev = -1
                for p in pred_seq:
                    if p != 0 and p != prev: clean_pred.append(p)
                    prev = p
                    
                try: hyp_text = tokenizer.decode(clean_pred)
                except: hyp_text = "".join([str(c) for c in clean_pred])
                
                cer, wer = calculate_cer(ref_text, hyp_text), calculate_wer(ref_text, hyp_text)
                total_cer += cer
                total_wer += wer
                
                is_exact = (ref_text == hyp_text)
                if is_exact: exact_matches += 1
                
                tgt_len = len(ref_text)
                prd_len = len(hyp_text)
                target_length_sum += tgt_len
                pred_length_sum += prd_len
                
                bucket = get_bucket(tgt_len)
                buckets[bucket]["count"] += 1
                buckets[bucket]["target_len_sum"] += tgt_len
                buckets[bucket]["pred_len_sum"] += prd_len
                buckets[bucket]["cer_sum"] += cer
                if is_exact: buckets[bucket]["exact_matches"] += 1
                
                num_samples += 1
                
            if num_samples % 5000 < 128:
                print(f"Processed {num_samples} samples...")

    avg_cer = total_cer / num_samples if num_samples > 0 else 0
    avg_wer = total_wer / num_samples if num_samples > 0 else 0
    char_acc = max(0.0, 1.0 - avg_cer)
    word_acc = max(0.0, 1.0 - avg_wer)
    blank_percentage = (blank_preds / total_raw_preds * 100) if total_raw_preds > 0 else 100.0
    avg_target_len = target_length_sum / num_samples if num_samples > 0 else 0
    avg_pred_len = pred_length_sum / num_samples if num_samples > 0 else 0
    
    print("\n=== OVERALL TEST METRICS ===")
    print(f"Samples: {num_samples}")
    print(f"CER: {avg_cer:.4f}")
    print(f"WER: {avg_wer:.4f}")
    print(f"Character Accuracy: {char_acc*100:.2f}%")
    print(f"Word Accuracy: {word_acc*100:.2f}%")
    print(f"Exact Matches: {exact_matches}")
    print(f"Blank %: {blank_percentage:.2f}%")
    print(f"Average Target Length: {avg_target_len:.2f}")
    print(f"Average Prediction Length: {avg_pred_len:.2f}")
    
    print("\n=== BUCKET METRICS ===")
    for b in ["1-5", "6-10", "11-15", "16+"]:
        cnt = buckets[b]["count"]
        if cnt == 0: continue
        avg_tgt = buckets[b]["target_len_sum"] / cnt
        avg_prd = buckets[b]["pred_len_sum"] / cnt
        cer = buckets[b]["cer_sum"] / cnt
        exact = buckets[b]["exact_matches"] / cnt * 100
        
        print(f"Bucket {b}:")
        print(f"  Count: {cnt}")
        print(f"  Avg Target Len: {avg_tgt:.2f}")
        print(f"  Avg Pred Len: {avg_prd:.2f}")
        print(f"  CER: {cer:.4f}")
        print(f"  Exact Accuracy: {exact:.2f}%")

if __name__ == "__main__":
    main()
