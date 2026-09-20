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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode
from scripts.dataset_loader import get_dataloader
from models.digitalization.tokenizer import Tokenizer
import editdistance

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

def main():
    parser = argparse.ArgumentParser(description="Reusable OCR Evaluation Tool")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, required=True, help="Dataset split (e.g., validation, test)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--manifest", type=str, default=None, help="Path to fixed test set manifest (optional)")
    parser.add_argument("--less_downsample", action="store_true", help="Use less_downsample for CRNN")
    args = parser.parse_args()

    working_root = "data/tamil_ocr_dataset"
    vocab_path = os.path.join(working_root, "vocabulary", "tamil_vocab.json")
    
    tokenizer = Tokenizer(vocab_path=vocab_path)
    model = CRNN(tokenizer.num_classes, feature_size=2048, less_downsample=args.less_downsample).to(args.device)
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.manifest:
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms
        from PIL import Image
        import numpy as np

        class FixedSubset(Dataset):
            def __init__(self, manifest, tokenizer):
                with open(manifest, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.samples = data.get("samples", [])
                self.tokenizer = tokenizer
                self.transform = transforms.Compose([
                    transforms.Resize((32, 100)), # Simplified
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
                from scripts.dataset_loader import AspectRatioPreservingResize
                self.transform = transforms.Compose([
                    AspectRatioPreservingResize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
                
            def __len__(self): return len(self.samples)
            
            def __getitem__(self, idx):
                item = self.samples[idx]
                img_path = Path("C:/Users/prsan/Desktop/CDAC/1_Draft/data/tamil_ocr_dataset/imported/tamil") / item["split"] / item["packet"] / item["image"]
                try: img = Image.open(img_path).convert('L')
                except: img = Image.new('L', (100, 32), 255)
                img = self.transform(img)
                target = self.tokenizer.encode(item["label"])
                return img, torch.tensor(target, dtype=torch.long), len(target), img_path

        from scripts.dataset_loader import crnn_collate_fn
        dataset = FixedSubset(args.manifest, tokenizer)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=crnn_collate_fn)
    else:
        dataloader = get_dataloader(
            dataset_root=working_root,
            language="tamil",
            split=args.split,
            tokenizer=tokenizer,
            batch_size=128,
            num_workers=0,
            max_packets=None,
            target_packet=None,
            replay_previous=False,
            shuffle=False
        )
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    val_loss = 0.0
    total_cer, total_wer = 0.0, 0.0
    num_samples, total_raw_preds, blank_preds, exact_matches = 0, 0, 0, 0
    all_predicted_chars = []
    unique_words = set()

    start_time = time.time()
    with torch.no_grad():
        for batch_data in dataloader:
            if len(batch_data) == 5:
                images, targets, target_lengths, actual_widths, _ = batch_data
            else:
                images, targets, target_lengths, actual_widths = batch_data
            
            images = images.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)
            
            outputs = model(images)
            outputs_perm = outputs.permute(1, 0, 2)
            outputs_log_probs = outputs_perm.log_softmax(2)
            
            input_lengths = torch.round((actual_widths.float().to(args.device) / images.size(3)) * outputs_perm.size(0)).to(torch.long)
            input_lengths = torch.clamp(input_lengths, max=outputs_perm.size(0))
            
            loss = criterion(outputs_log_probs, targets, input_lengths, target_lengths)
            val_loss += loss.item()
            
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
                if ref_text == hyp_text: exact_matches += 1
                
                for c in hyp_text: all_predicted_chars.append(c)
                unique_words.add(hyp_text)
                num_samples += 1

    eval_time = time.time() - start_time
    avg_loss = val_loss / len(dataloader)
    avg_cer = total_cer / num_samples if num_samples > 0 else 0
    avg_wer = total_wer / num_samples if num_samples > 0 else 0
    char_acc = max(0.0, 1.0 - avg_cer)
    word_acc = max(0.0, 1.0 - avg_wer)
    
    char_counts = Counter(all_predicted_chars)
    blank_percentage = (blank_preds / total_raw_preds * 100) if total_raw_preds > 0 else 100.0
    avg_pred_length = (sum(char_counts.values()) / num_samples) if num_samples > 0 else 0.0

    metrics = {
        "CER": round(avg_cer, 4),
        "WER": round(avg_wer, 4),
        "Character Accuracy": round(char_acc, 4),
        "Word Accuracy": round(word_acc, 4),
        "Blank Percentage": round(blank_percentage, 2),
        "Unique Predicted Characters": len(char_counts),
        "Unique Predicted Sequences": len(unique_words),
        "Average Prediction Length": round(avg_pred_length, 2),
        "Validation Loss": round(avg_loss, 4),
        "Exact Match Accuracy": round(exact_matches / num_samples if num_samples > 0 else 0, 4)
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    for k, v in metrics.items(): print(f"{k}: {v}")
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
