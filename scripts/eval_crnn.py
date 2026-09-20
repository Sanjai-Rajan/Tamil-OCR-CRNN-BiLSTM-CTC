import sys
import os
import argparse
import torch
from pathlib import Path

# Force UTF-8 encoding for standard output on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode
from scripts.dataset_loader import get_dataloader
from models.digitalization.tokenizer import Tokenizer
from utils.device import get_device
from utils.metrics import calculate_cer, calculate_wer

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--language", type=str, default="tamil")
    parser.add_argument("--max-packets", type=int, default=None, help="Max packets to load (None for all)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    working_root = "data/tamil_ocr_dataset"
    vocab_path = os.path.join(working_root, "vocabulary", f"{args.language}_vocab.json")
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    is_temporal_fix = 'temporal_fix' in args.checkpoint
    model = CRNN(tokenizer.num_classes, feature_size=2048, less_downsample=is_temporal_fix).to(device)
    
    print(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Handle state_dict loading
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    print("Loading test dataset...")
    dataloader = get_dataloader(
        dataset_root=working_root, 
        language=args.language, 
        split="test", 
        tokenizer=tokenizer, 
        batch_size=args.batch_size, 
        max_packets=args.max_packets, 
        shuffle=False
    )
    
    total_cer = 0
    total_wer = 0
    num_samples = 0
    total_raw_preds = 0
    blank_preds = 0
    
    exact_word_matches = 0
    total_pred_len = 0
    total_target_len = 0
    
    # For Phase 3 Analysis
    length_groups = {
        "1-5": {"count": 0, "target_len": 0, "pred_len": 0, "cer": 0, "exact": 0, "examples": []},
        "6-10": {"count": 0, "target_len": 0, "pred_len": 0, "cer": 0, "exact": 0, "examples": []},
        "11-15": {"count": 0, "target_len": 0, "pred_len": 0, "cer": 0, "exact": 0, "examples": []},
        "16+": {"count": 0, "target_len": 0, "pred_len": 0, "cer": 0, "exact": 0, "examples": []},
    }

    print("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if len(batch_data) == 5:
                images, targets, target_lengths, actual_widths, _ = batch_data
            else:
                images, targets, target_lengths, actual_widths = batch_data
                
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            preds = ctc_decode(outputs)
            
            total_raw_preds += preds.numel()
            blank_preds += (preds == 0).sum().item()
            
            offset = 0
            for i in range(len(target_lengths)):
                length = target_lengths[i].item()
                target_seq = targets[offset:offset+length].cpu().tolist()
                offset += length
                
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
                    
                try:
                    hyp_text = tokenizer.decode(clean_pred)
                except AttributeError:
                    hyp_text = "".join([str(c) for c in clean_pred])
                    
                cer = calculate_cer(ref_text, hyp_text)
                wer = calculate_wer(ref_text, hyp_text)
                total_cer += cer
                total_wer += wer
                num_samples += 1
                
                t_len = len(ref_text)
                p_len = len(hyp_text)
                total_target_len += t_len
                total_pred_len += p_len
                
                is_exact = (ref_text == hyp_text)
                if is_exact:
                    exact_word_matches += 1
                    
                if t_len <= 5:
                    grp = "1-5"
                elif t_len <= 10:
                    grp = "6-10"
                elif t_len <= 15:
                    grp = "11-15"
                else:
                    grp = "16+"
                    
                length_groups[grp]["count"] += 1
                length_groups[grp]["target_len"] += t_len
                length_groups[grp]["pred_len"] += p_len
                length_groups[grp]["cer"] += cer
                if is_exact:
                    length_groups[grp]["exact"] += 1
                
                # Keep up to 10 examples per group (prefer errors to inspect truncation)
                if len(length_groups[grp]["examples"]) < 10 and not is_exact:
                    length_groups[grp]["examples"].append((ref_text, hyp_text))
                
            if batch_idx % 50 == 0:
                print(f"  Processed {num_samples} samples...")
                
    if num_samples == 0:
        print("No samples processed.")
        return
        
    avg_cer = total_cer / num_samples
    avg_wer = total_wer / num_samples
    blank_perc = (blank_preds / total_raw_preds * 100) if total_raw_preds > 0 else 100.0
    avg_target_len = total_target_len / num_samples
    avg_pred_len = total_pred_len / num_samples
    exact_match_perc = (exact_word_matches / num_samples) * 100.0
    
    print("\n============================================================")
    print("TEST EVALUATION RESULTS")
    print("============================================================")
    print(f"Total Test Samples: {num_samples}")
    print(f"CER: {avg_cer:.4f}")
    print(f"WER: {avg_wer:.4f}")
    print(f"Character Accuracy: {max(0, 1.0 - avg_cer):.4f}")
    print(f"Word Accuracy: {max(0, 1.0 - avg_wer):.4f}")
    print(f"Average prediction length: {avg_pred_len:.2f}")
    print(f"Average target length: {avg_target_len:.2f}")
    print(f"Blank Percentage: {blank_perc:.2f}%")
    print(f"Exact word matches: {exact_word_matches}")
    print(f"Exact word-match percentage: {exact_match_perc:.2f}%")
    print("============================================================")
    
    print("\n============================================================")
    print("LONG-WORD ANALYSIS")
    print("============================================================")
    for grp, stats in length_groups.items():
        c = stats["count"]
        if c == 0:
            continue
        print(f"\nGroup: {grp} tokens (Samples: {c})")
        print(f"  Avg Target Len: {stats['target_len']/c:.2f}")
        print(f"  Avg Pred Len  : {stats['pred_len']/c:.2f}")
        print(f"  CER           : {stats['cer']/c:.4f}")
        print(f"  Exact Match   : {stats['exact']/c*100:.2f}%")
        print("  Examples:")
        for ref, hyp in stats["examples"]:
            print(f"    TARGET: {ref}  ->  PREDICTION: {hyp}")
    print("============================================================")

if __name__ == "__main__":
    evaluate()
