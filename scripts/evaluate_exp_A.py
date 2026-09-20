import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import torch
import json
import time
from pathlib import Path
from PIL import Image
from torchvision import transforms

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from models.recognition.crnn import CRNN
from models.digitalization.tokenizer import Tokenizer
from scripts.dataset_loader import AspectRatioPreservingResize, get_dataloader
from models.recognition.decoder import ctc_decode
from collections import defaultdict
import Levenshtein

def evaluate_loader(model, loader, tokenizer, device):
    model.eval()
    total_cer, total_wer = 0.0, 0.0
    char_correct, char_total = 0, 0
    word_correct, word_total = 0, 0
    total_loss = 0.0
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    num_samples = 0
    exact_matches = 0

    with torch.no_grad():
        for i, (images, targets, target_lengths, widths, img_paths) in enumerate(loader):
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            
            # Loss calculation
            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
            input_lengths = torch.full((batch_size,), outputs.size(1), dtype=torch.long, device=device)
            loss = criterion(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)
            total_loss += loss.item() * batch_size

            preds = ctc_decode(outputs)

            start_idx = 0
            for j in range(batch_size):
                t_len = target_lengths[j].item()
                target_seq = targets[start_idx:start_idx+t_len].cpu().tolist()
                start_idx += t_len
                pred_seq = preds[j].cpu().tolist()
                
                clean_pred = []
                prev = -1
                for p in pred_seq:
                    if p != 0 and p != prev:
                        clean_pred.append(p)
                    prev = p

                try:
                    target_text = tokenizer.decode(target_seq)
                    pred_text = tokenizer.decode(clean_pred)
                except AttributeError:
                    target_text = "".join([str(c) for c in target_seq])
                    pred_text = "".join([str(c) for c in clean_pred])

                dist = Levenshtein.distance(target_text, pred_text)
                total_cer += dist
                char_correct += max(0, len(target_text) - dist)
                char_total += len(target_text)
                
                if target_text == pred_text:
                    exact_matches += 1

                t_words = target_text.split()
                p_words = pred_text.split()
                
                if len(t_words) > 0:
                    t_str = " ".join(t_words)
                    p_str = " ".join(p_words)
                    word_dist = Levenshtein.distance(t_words, p_words)
                    total_wer += word_dist
                    word_correct += max(0, len(t_words) - word_dist)
                    word_total += len(t_words)
                    
            num_samples += batch_size
            
            # Print progress for every 50 batches
            if (i+1) % 50 == 0:
                print(f"Eval batch {i+1}/{len(loader)}...")

    return {
        'cer': total_cer / max(char_total, 1),
        'wer': total_wer / max(word_total, 1),
        'char_acc': char_correct / max(char_total, 1),
        'word_acc': word_correct / max(word_total, 1),
        'exact_match': exact_matches / max(num_samples, 1),
        'loss': total_loss / max(num_samples, 1)
    }

def run_inference(model, image, transform, tokenizer, device):
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        preds = ctc_decode(outputs)
        pred_seq = preds[0].cpu().tolist()
        
        clean_pred = []
        prev = -1
        for p in pred_seq:
            if p != 0 and p != prev:
                clean_pred.append(p)
            prev = p
            
        try:
            pred_text = tokenizer.decode(clean_pred)
        except AttributeError:
            pred_text = "".join([str(c) for c in clean_pred])
    return pred_text

def get_model(ckpt_path, tokenizer, device):
    model = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_path = "data/tamil_ocr_dataset/vocabulary/tamil_vocab.json"
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    baseline_ckpt = "checkpoints/recognition/tamil/tamil_full_40epoch/best.pth"
    expA_ckpt = "checkpoints/recognition/tamil/exp_A_multiscale/best.pth"
    
    print("Loading Baseline model...")
    baseline_model = get_model(baseline_ckpt, tokenizer, device)
    
    print("Loading Exp A model...")
    expA_model = get_model(expA_ckpt, tokenizer, device)
    
    print("\n==================================================")
    print("1. VALIDATION METRICS")
    print("==================================================")
    # Get val loader
    val_loader = get_dataloader(
        dataset_root="data/tamil_ocr_dataset",
        language="tamil",
        split='validation',
        tokenizer=tokenizer,
        batch_size=32,
        num_workers=4,
        max_packets=None,
        replay_previous=False,
        shuffle=False,
        use_bucketing=False,
        use_multiscale_aug=False
    )
    
    print("Evaluating Baseline...")
    b_metrics = evaluate_loader(baseline_model, val_loader, tokenizer, device)
    print("Evaluating Exp A...")
    a_metrics = evaluate_loader(expA_model, val_loader, tokenizer, device)
    
    print(f"\nMetric | Baseline | Exp A | Absolute Change | Relative Change")
    print("-" * 75)
    for key, name in [('cer', 'CER'), ('wer', 'WER'), ('char_acc', 'Character Accuracy'), 
                      ('word_acc', 'Word Accuracy'), ('exact_match', 'Exact Match Accuracy'), ('loss', 'Validation Loss')]:
        b_val = b_metrics[key]
        a_val = a_metrics[key]
        abs_change = a_val - b_val
        rel_change = (abs_change / b_val * 100) if b_val != 0 else 0
        print(f"{name} | {b_val:.4f} | {a_val:.4f} | {abs_change:+.4f} | {rel_change:+.2f}%")
        
    print("\n==================================================")
    print("2. SCALE ROBUSTNESS TEST")
    print("==================================================")
    
    manifest_path = "data/tamil_ocr_dataset/imported/tamil/train/packet_001/manifest.jsonl"
    target_data, orig_img, ground_truth, orig_w, orig_h, img_path = None, None, "", 0, 0, ""
    
    transform = transforms.Compose([
        AspectRatioPreservingResize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            p = os.path.join("data/tamil_ocr_dataset/imported/tamil/train/packet_001", data["image"])
            try:
                img = Image.open(p).convert('L')
                pred = run_inference(baseline_model, img, transform, tokenizer, device)
                if pred == data["label"]:
                    target_data = data
                    orig_img = img
                    ground_truth = data["label"]
                    orig_w, orig_h = img.size
                    img_path = p
                    break
            except Exception as e:
                pass
                
    print(f"Selected Image: {img_path}")
    print(f"Ground Truth: {ground_truth}")
    print("\nScale | Baseline Prediction | Baseline Correct? | Exp A Prediction | Exp A Correct?")
    print("-" * 100)
    
    scales = [("100%", 1.0), ("75%", 0.75), ("50%", 0.50), ("37.5%", 0.375), ("25%", 0.25)]
    for scale_name, scale_factor in scales:
        new_w = max(1, int(orig_w * scale_factor))
        new_h = max(1, int(orig_h * scale_factor))
        shrunk_img = orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new('L', (orig_w, orig_h), color=255)
        offset_x = (orig_w - new_w) // 2
        offset_y = (orig_h - new_h) // 2
        canvas.paste(shrunk_img, (offset_x, offset_y))
        
        b_pred = run_inference(baseline_model, canvas, transform, tokenizer, device)
        a_pred = run_inference(expA_model, canvas, transform, tokenizer, device)
        
        b_corr = "YES" if b_pred == ground_truth else "NO"
        a_corr = "YES" if a_pred == ground_truth else "NO"
        
        print(f"{scale_name:<5} | {b_pred:<20} | {b_corr:<17} | {a_pred:<18} | {a_corr}")
        
    print("\n==================================================")
    print("3. SAMPLE TEST")
    print("==================================================")
    sample_path = "sample_test.png"
    sample_img = Image.open(sample_path).convert('L')
    b_sample_pred = run_inference(baseline_model, sample_img, transform, tokenizer, device)
    a_sample_pred = run_inference(expA_model, sample_img, transform, tokenizer, device)
    print(f"Baseline:\n{b_sample_pred}\n")
    print(f"Exp A:\n{a_sample_pred}")
    
    print("\n==================================================")
    print("4. TRAINING ANALYSIS")
    print("==================================================")
    # Read training history
    history_file = "outputs/training/tamil/exp_A_multiscale/training_history.csv"
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip().split(',')
                headers = lines[0].strip().split(',')
                h_dict = dict(zip(headers, last_line))
                
                print(f"Total training time: {h_dict.get('training_time', 'N/A')}s (partial/accumulated)")
                print(f"GPU used: cuda")
                print(f"Best epoch: {h_dict.get('epoch', 'N/A')}")
                print(f"Final training loss: {h_dict.get('train_loss', 'N/A')}")
                print(f"Final validation loss: {h_dict.get('validation_loss', 'N/A')}")
                print(f"Best validation metric (CER): {h_dict.get('CER', 'N/A')}")
                
                t_loss = float(h_dict.get('train_loss', 0))
                v_loss = float(h_dict.get('validation_loss', 0))
                overfit = "YES" if v_loss > (t_loss * 1.5) else "NO"
                print(f"Overfitting occurred: {overfit}")
    else:
        print("Training history file not found.")
        
    print(f"Checkpoint path: {expA_ckpt}")

if __name__ == "__main__":
    main()
