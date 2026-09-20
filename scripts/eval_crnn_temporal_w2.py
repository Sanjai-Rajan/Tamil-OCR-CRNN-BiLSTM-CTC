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
import matplotlib.pyplot as plt
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Redefine W/2 architecture for eval
from models.recognition.lstm import SequenceModel
from models.recognition.head import PredictionHead
from scripts.dataset_loader import get_dataloader
from models.digitalization.tokenizer import Tokenizer

class EncoderW2(nn.Module):
    def __init__(self, less_downsample=False):
        super().__init__()
        pool1 = nn.MaxPool2d((2, 2))
        pool2 = nn.MaxPool2d((2, 1))
        pool3 = nn.MaxPool2d((2, 1))
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), pool1,
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), pool2,
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), pool3,
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        )
    def forward(self, x): return self.features(x)

class CRNNW2(nn.Module):
    def __init__(self, num_classes, feature_size=2048, less_downsample=False):
        super().__init__()
        self.encoder = EncoderW2(less_downsample=less_downsample)
        self.sequence = SequenceModel(input_size=feature_size)
        self.head = PredictionHead(num_classes)
    def forward(self, x):
        x = self.encoder(x)
        batch, channel, height, width = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, width, channel * height)
        x = self.sequence(x)
        x = self.head(x)
        return x

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

def ctc_decode(log_probs, blank=0):
    probs = log_probs.softmax(dim=-1)
    argmax = probs.argmax(dim=-1)
    return argmax

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoints/recognition/tamil/ocr_temporal_resolution_w2/best.pth"
    working_root = "data/tamil_ocr_dataset"
    vocab_path = os.path.join(working_root, "vocabulary", "tamil_vocab.json")
    
    tokenizer = Tokenizer(vocab_path=vocab_path)
    model = CRNNW2(tokenizer.num_classes, feature_size=2048, less_downsample=True).to(device)
    
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
    total_required_t = 0
    total_ctc_t = 0
    total_margin = 0
    total_nonblank_t = 0
    
    buckets = {
        "1-5": {"count": 0, "target_len_sum": 0, "pred_len_sum": 0, "cer_sum": 0, "wer_sum": 0, "exact_matches": 0, "blank_preds": 0, "raw_preds": 0},
        "6-10": {"count": 0, "target_len_sum": 0, "pred_len_sum": 0, "cer_sum": 0, "wer_sum": 0, "exact_matches": 0, "blank_preds": 0, "raw_preds": 0},
        "11-15": {"count": 0, "target_len_sum": 0, "pred_len_sum": 0, "cer_sum": 0, "wer_sum": 0, "exact_matches": 0, "blank_preds": 0, "raw_preds": 0},
        "16+": {"count": 0, "target_len_sum": 0, "pred_len_sum": 0, "cer_sum": 0, "wer_sum": 0, "exact_matches": 0, "blank_preds": 0, "raw_preds": 0},
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
            
            # calculate actual T per sample
            input_lengths = torch.round((actual_widths.float() / images.size(3)) * outputs.size(1)).to(torch.long)
            
            offset = 0
            for i in range(len(target_lengths)):
                length = target_lengths[i].item()
                target_seq = targets[offset:offset+length].cpu().tolist()
                offset += length
                
                try: ref_text = tokenizer.decode(target_seq)
                except: ref_text = "".join([str(c) for c in target_seq])
                    
                T_i = input_lengths[i].item()
                pred_seq = preds[i, :T_i].cpu().tolist()
                
                blanks_i = pred_seq.count(0)
                raw_i = len(pred_seq)
                
                blank_preds += blanks_i
                total_raw_preds += raw_i
                
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
                
                req_T = tgt_len + sum(1 for j in range(tgt_len - 1) if target_seq[j] == target_seq[j+1])
                total_required_t += req_T
                total_ctc_t += T_i
                total_margin += (T_i - req_T)
                total_nonblank_t += (raw_i - blanks_i)
                
                bucket = get_bucket(tgt_len)
                buckets[bucket]["count"] += 1
                buckets[bucket]["target_len_sum"] += tgt_len
                buckets[bucket]["pred_len_sum"] += prd_len
                buckets[bucket]["cer_sum"] += cer
                buckets[bucket]["wer_sum"] += wer
                buckets[bucket]["blank_preds"] += blanks_i
                buckets[bucket]["raw_preds"] += raw_i
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
    
    out_dir = Path("outputs/training/benchmarks/temporal_resolution_w2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Baseline W/4 metrics
    w4_baseline = {
        "overall": {
            "cer": 0.2176,
            "wer": 0.4199,
            "char_acc": 0.7824,
            "word_acc": 0.5801,
            "target_len": 4.79,
            "pred_len": 3.26,
            "blank_pct": 92.97
        },
        "buckets": {
            "1-5": {"cer": 0.0696, "word_acc": 0.8730, "target_avg": 1.97, "pred_avg": 1.99},
            "6-10": {"cer": 0.4534, "word_acc": 0.0052, "target_avg": 8.13, "pred_avg": 5.53},
            "11-15": {"cer": 0.5720, "word_acc": 0.0000, "target_avg": 12.57, "pred_avg": 5.98},
            "16+": {"cer": 0.6663, "word_acc": 0.0000, "target_avg": 17.92, "pred_avg": 6.38}
        }
    }

    w2_metrics = {
        "overall": {
            "cer": avg_cer,
            "wer": avg_wer,
            "char_acc": char_acc,
            "word_acc": word_acc,
            "target_len": avg_target_len,
            "pred_len": avg_pred_len,
            "blank_pct": blank_percentage,
            "mean_ctc_t": total_ctc_t / num_samples,
            "mean_required_t": total_required_t / num_samples,
            "mean_temporal_margin": total_margin / num_samples,
            "mean_nonblank_t": total_nonblank_t / num_samples
        },
        "buckets": {}
    }
    
    with open(out_dir / "w2_length_metrics.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Bucket", "Count", "Avg Target", "Avg Pred", "Ratio", "CER", "WER", "Exact Match %", "Blank %"])
        for b in ["1-5", "6-10", "11-15", "16+"]:
            cnt = buckets[b]["count"]
            if cnt == 0: continue
            avg_tgt = buckets[b]["target_len_sum"] / cnt
            avg_prd = buckets[b]["pred_len_sum"] / cnt
            cer = buckets[b]["cer_sum"] / cnt
            wer = buckets[b]["wer_sum"] / cnt
            exact = buckets[b]["exact_matches"] / cnt
            b_pct = buckets[b]["blank_preds"] / buckets[b]["raw_preds"] * 100 if buckets[b]["raw_preds"] > 0 else 100
            
            w2_metrics["buckets"][b] = {
                "cer": cer,
                "word_acc": exact,
                "target_avg": avg_tgt,
                "pred_avg": avg_prd,
                "blank_pct": b_pct
            }
            writer.writerow([b, cnt, avg_tgt, avg_prd, avg_prd/avg_tgt, cer, wer, exact*100, b_pct])

    with open(out_dir / "w2_metrics.json", "w") as f:
        json.dump(w2_metrics, f, indent=4)
        
    with open(out_dir / "w4_vs_w2_comparison.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "W/4 Baseline", "W/2 Experiment", "Diff"])
        writer.writerow(["CER", w4_baseline["overall"]["cer"], avg_cer, avg_cer - w4_baseline["overall"]["cer"]])
        writer.writerow(["WER", w4_baseline["overall"]["wer"], avg_wer, avg_wer - w4_baseline["overall"]["wer"]])
        writer.writerow(["Char Acc", w4_baseline["overall"]["char_acc"], char_acc, char_acc - w4_baseline["overall"]["char_acc"]])
        writer.writerow(["Word Acc", w4_baseline["overall"]["word_acc"], word_acc, word_acc - w4_baseline["overall"]["word_acc"]])
        writer.writerow(["Pred Len", w4_baseline["overall"]["pred_len"], avg_pred_len, avg_pred_len - w4_baseline["overall"]["pred_len"]])
        writer.writerow(["Blank %", w4_baseline["overall"]["blank_pct"], blank_percentage, blank_percentage - w4_baseline["overall"]["blank_pct"]])

    # Plot 1: Overall CER/WER
    metrics = ['CER', 'WER']
    w4_vals = [w4_baseline["overall"]["cer"], w4_baseline["overall"]["wer"]]
    w2_vals = [avg_cer, avg_wer]
    x = range(len(metrics))
    plt.figure()
    plt.bar([i - 0.2 for i in x], w4_vals, 0.4, label='W/4')
    plt.bar([i + 0.2 for i in x], w2_vals, 0.4, label='W/2')
    plt.xticks(x, metrics)
    plt.legend()
    plt.title("W/4 vs W/2 Overall Error Rates")
    plt.savefig(out_dir / "1_overall_cer_wer.png")
    
    # Plot 2: Average Prediction Length
    plt.figure()
    plt.bar(['W/4', 'W/2'], [w4_baseline["overall"]["pred_len"], avg_pred_len])
    plt.title("Average Prediction Length (Target = 4.79)")
    plt.savefig(out_dir / "2_avg_pred_length.png")
    
    # Plot 3: Target vs Prediction by bucket
    buckets_list = ["1-5", "6-10", "11-15", "16+"]
    w4_preds = [w4_baseline["buckets"][b]["pred_avg"] for b in buckets_list]
    w2_preds = [w2_metrics["buckets"][b]["pred_avg"] for b in buckets_list]
    targets = [w4_baseline["buckets"][b]["target_avg"] for b in buckets_list]
    
    x = range(4)
    plt.figure()
    plt.plot(x, targets, 'ko-', label='Target')
    plt.plot(x, w4_preds, 'ro-', label='W/4 Pred')
    plt.plot(x, w2_preds, 'bo-', label='W/2 Pred')
    plt.xticks(x, buckets_list)
    plt.legend()
    plt.title("Target vs Predicted Length by Bucket")
    plt.savefig(out_dir / "3_target_vs_pred_by_bucket.png")
    
    # Plot 4: Blank %
    plt.figure()
    plt.bar(['W/4', 'W/2'], [w4_baseline["overall"]["blank_pct"], blank_percentage])
    plt.title("Overall Blank Percentage")
    plt.savefig(out_dir / "4_blank_percentage.png")
    
    # Plot 5: CER by bucket
    w4_cer = [w4_baseline["buckets"][b]["cer"] for b in buckets_list]
    w2_cer = [w2_metrics["buckets"][b]["cer"] for b in buckets_list]
    plt.figure()
    plt.bar([i - 0.2 for i in x], w4_cer, 0.4, label='W/4')
    plt.bar([i + 0.2 for i in x], w2_cer, 0.4, label='W/2')
    plt.xticks(x, buckets_list)
    plt.legend()
    plt.title("CER by Length Bucket")
    plt.savefig(out_dir / "5_cer_by_bucket.png")
    
    # Plot 6: Word Acc by bucket
    w4_acc = [w4_baseline["buckets"][b]["word_acc"] for b in buckets_list]
    w2_acc = [w2_metrics["buckets"][b]["word_acc"] for b in buckets_list]
    plt.figure()
    plt.bar([i - 0.2 for i in x], w4_acc, 0.4, label='W/4')
    plt.bar([i + 0.2 for i in x], w2_acc, 0.4, label='W/2')
    plt.xticks(x, buckets_list)
    plt.legend()
    plt.title("Word Accuracy by Length Bucket")
    plt.savefig(out_dir / "6_word_acc_by_bucket.png")
    
    # Plot 7: Pred/Target Ratio
    w4_ratio = [w4_preds[i]/targets[i] for i in range(4)]
    w2_ratio = [w2_preds[i]/targets[i] for i in range(4)]
    plt.figure()
    plt.bar([i - 0.2 for i in x], w4_ratio, 0.4, label='W/4')
    plt.bar([i + 0.2 for i in x], w2_ratio, 0.4, label='W/2')
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.xticks(x, buckets_list)
    plt.legend()
    plt.title("Prediction / Target Length Ratio")
    plt.savefig(out_dir / "7_pred_target_ratio.png")

    print("Evaluation completed.")

if __name__ == "__main__":
    main()
