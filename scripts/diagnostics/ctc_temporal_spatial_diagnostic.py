import sys
import os
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import editdistance
from pathlib import Path

sys.path.insert(0, '.')
from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode
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
    checkpoint_path = r"checkpoints\recognition\tamil\ocr_balanced_temporal\best.pth"
    vocab_path = r"data\tamil_ocr_dataset\vocabulary\tamil_vocab.json"
    manifest_path = r"data\tamil_ocr_dataset\imported\tamil\test\packet_001\manifest.jsonl"
    img_dir = r"data\tamil_ocr_dataset\imported\tamil\test\packet_001"
    
    out_dir = Path(r"outputs\diagnostics\ctc_temporal_spatial_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = Tokenizer(vocab_path=vocab_path)
    blank_idx = getattr(tokenizer, 'blank_idx', 0)
    
    model = CRNN(tokenizer.num_classes, feature_size=2048, less_downsample=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Stratified sampling
    random.seed(42)
    samples_by_bucket = {"1-5": [], "6-10": [], "11-15": [], "16+": []}
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            tokens = tokenizer.encode(sample["label"])
            bucket = get_bucket(len(tokens))
            samples_by_bucket[bucket].append((sample, tokens))
            
    # Sample up to 50 from each bucket
    selected_samples = []
    for b in ["1-5", "6-10", "11-15", "16+"]:
        pool = samples_by_bucket[b]
        random.shuffle(pool)
        selected_samples.extend(pool[:50])
        
    print(f"Selected {len(selected_samples)} samples for diagnostic.")
    
    results = []
    representative_samples = {"1-5": None, "6-10": None, "11-15": None, "16+": None}
    
    for idx, (sample, target_seq) in enumerate(selected_samples):
        img_path = os.path.join(img_dir, sample["image"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        orig_img = cv2.imread(img_path)
        
        h, w = img.shape
        target_h = 32
        ratio = target_h / float(h)
        target_w = int(w * ratio)
        img_resized = cv2.resize(img, (target_w, target_h))
        
        img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        img_tensor = img_tensor.sub_(0.5).div_(0.5).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            # CRNN output is [batch, width, num_classes]
            T = outputs.shape[1]
            
            preds = ctc_decode(outputs)
            pred_seq = preds[0].cpu().tolist()
            
            clean_pred = []
            prev = -1
            for p in pred_seq:
                if p != blank_idx and p != prev:
                    clean_pred.append(p)
                prev = p
                
            probs = torch.softmax(outputs[0, :, :], dim=-1) # [T, num_classes]
            argmax = probs.argmax(dim=-1) # [T]
            max_probs, _ = probs.max(dim=-1) # [T]
            
        target_len = len(target_seq)
        pred_len = len(clean_pred)
        
        adjacent_repeats = sum(1 for i in range(target_len - 1) if target_seq[i] == target_seq[i+1])
        required_T = target_len + adjacent_repeats
        temporal_margin = T - required_T
        
        blank_mask = (argmax == blank_idx)
        blank_percentage = blank_mask.float().mean().item() * 100
        nonblank_timesteps = (~blank_mask).sum().item()
        
        ref_text = sample["label"]
        try:
            pred_text = tokenizer.decode(clean_pred)
        except:
            pred_text = "".join([str(x) for x in clean_pred])
            
        cer = calculate_cer(ref_text, pred_text)
        wer = calculate_wer(ref_text, pred_text)
        exact = 1 if cer == 0 else 0
        
        ratio_len = pred_len / target_len if target_len > 0 else 0
        bucket = get_bucket(target_len)
        
        res_dict = {
            "sample_id": idx,
            "image_path": sample["image"],
            "target_text": ref_text,
            "prediction": pred_text,
            "target_length": target_len,
            "prediction_length": pred_len,
            "image_width": w,
            "image_height": h,
            "feature_width": T,
            "ctc_T": T,
            "required_T": required_T,
            "temporal_margin": temporal_margin,
            "blank_percentage": blank_percentage,
            "nonblank_timesteps": nonblank_timesteps,
            "prediction_target_ratio": ratio_len,
            "CER": cer,
            "WER": wer,
            "exact_match": exact,
            "bucket": bucket
        }
        results.append(res_dict)
        
        is_failure = ratio_len < 0.6 and target_len >= 6
        if is_failure and representative_samples[bucket] is None:
            representative_samples[bucket] = (res_dict, orig_img, argmax.cpu().numpy(), max_probs.cpu().numpy())
        elif representative_samples[bucket] is None and target_len <= 5:
            representative_samples[bucket] = (res_dict, orig_img, argmax.cpu().numpy(), max_probs.cpu().numpy())
            
    csv_keys = results[0].keys()
    with open(out_dir / "temporal_spatial_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys)
        writer.writeheader()
        for r in results: writer.writerow(r)
        
    buckets_list = ["1-5", "6-10", "11-15", "16+"]
    
    plt.figure(figsize=(8,6))
    plt.scatter([r["target_length"] for r in results], [r["prediction_length"] for r in results], alpha=0.5)
    plt.plot([0, 25], [0, 25], 'r--')
    plt.xlabel("Target Length")
    plt.ylabel("Predicted Length")
    plt.title("Target vs Predicted Length (Length Collapse)")
    plt.savefig(out_dir / "target_vs_predicted_length.png")
    plt.close()
    
    plt.figure(figsize=(8,6))
    plt.scatter([r["target_length"] for r in results], [r["ctc_T"] for r in results], label="Actual T", alpha=0.5)
    plt.scatter([r["target_length"] for r in results], [r["required_T"] for r in results], label="Required T", alpha=0.5)
    plt.xlabel("Target Length")
    plt.ylabel("Timesteps (T)")
    plt.legend()
    plt.title("Available Temporal Capacity (T) vs Required T")
    plt.savefig(out_dir / "target_vs_temporal_capacity.png")
    plt.close()
    
    bucket_stats = {}
    for b in buckets_list:
        b_res = [r for r in results if r["bucket"] == b]
        if not b_res: continue
        bucket_stats[b] = {
            "avg_margin": np.mean([r["temporal_margin"] for r in b_res]),
            "avg_blank_pct": np.mean([r["blank_percentage"] for r in b_res]),
            "avg_ratio": np.mean([r["prediction_target_ratio"] for r in b_res])
        }
    
    plt.figure(figsize=(8,6))
    plt.bar(bucket_stats.keys(), [v["avg_margin"] for v in bucket_stats.values()])
    plt.xlabel("Target Length Bucket")
    plt.ylabel("Average Temporal Margin (T - Required T)")
    plt.title("Temporal Margin by Length Bucket")
    plt.savefig(out_dir / "temporal_margin_by_bucket.png")
    plt.close()
    
    plt.figure(figsize=(8,6))
    plt.bar(bucket_stats.keys(), [v["avg_blank_pct"] for v in bucket_stats.values()])
    plt.xlabel("Target Length Bucket")
    plt.ylabel("Average Blank Percentage")
    plt.title("Blank Percentage by Length Bucket")
    plt.savefig(out_dir / "blank_percentage_by_bucket.png")
    plt.close()
    
    for b in buckets_list:
        if representative_samples[b] is None: continue
        res, orig_img, argmax, max_probs = representative_samples[b]
        
        T = res["ctc_T"]
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Target Length: {res['target_length']} | Pred Length: {res['prediction_length']} | T: {T}")
        plt.axis("off")
        
        plt.subplot(2, 1, 2)
        timeline = np.zeros((1, T))
        for t in range(T):
            if argmax[t] == blank_idx:
                timeline[0, t] = 0.2
            else:
                timeline[0, t] = 1.0
        plt.imshow(timeline, aspect="auto", cmap="magma", vmin=0, vmax=1)
        plt.yticks([])
        plt.xlabel("CTC Timestep")
        plt.title(f"Temporal Alignment (Grey = Blank, Bright = Token)\nMargin: {res['temporal_margin']} | Blank: {res['blank_percentage']:.1f}%")
        
        plt.tight_layout()
        plt.savefig(out_dir / f"alignment_vis_{b}.png")
        plt.close()

    print("Diagnostics complete.")

if __name__ == "__main__":
    main()
