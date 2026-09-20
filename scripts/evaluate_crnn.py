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
import matplotlib.pyplot as plt
from PIL import Image

# Force UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode
from scripts.dataset_loader import get_dataloader
from models.digitalization.tokenizer import Tokenizer
from utils.device import get_device
from utils.metrics import calculate_cer, calculate_wer

def load_config(config_path="config/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_contact_sheet(examples, output_path):
    """
    Generates a visual contact sheet (HTML) to robustly handle Tamil font rendering.
    Also attempts a Matplotlib version.
    """
    html_path = str(output_path).replace('.png', '.html')
    png_path = str(output_path).replace('.html', '.png')
    
    # HTML Version
    html_content = ["<html><head><meta charset='utf-8'><title>Validation Contact Sheet</title>"]
    html_content.append("<style>table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid black; padding: 8px; text-align: left;}</style>")
    html_content.append("</head><body><h2>Validation Contact Sheet</h2><table>")
    html_content.append("<tr><th>Image</th><th>Ground Truth</th><th>Prediction</th><th>CER</th><th>Category</th></tr>")
    
    for ex in examples:
        img_rel_path = os.path.relpath(ex['image_path'], start=os.path.dirname(html_path))
        cat = "Correct" if ex['cer'] == 0 else "Partial" if ex['cer'] <= 0.5 else "Empty" if not ex['pred'].strip() else "Severe"
        html_content.append(f"<tr>")
        html_content.append(f"<td><img src='{img_rel_path}' height='64'></td>")
        html_content.append(f"<td>{ex['gt']}</td>")
        html_content.append(f"<td>{ex['pred']}</td>")
        html_content.append(f"<td>{ex['cer']:.4f}</td>")
        html_content.append(f"<td>{cat}</td>")
        html_content.append(f"</tr>")
        
    html_content.append("</table></body></html>")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_content))
        
    # Matplotlib Version (Fallback, fonts might not render Tamil correctly)
    try:
        fig, axes = plt.subplots(len(examples), 1, figsize=(10, 2 * len(examples)))
        if len(examples) == 1:
            axes = [axes]
        
        for ax, ex in zip(axes, examples):
            img = Image.open(ex['image_path']).convert('L')
            ax.imshow(img, cmap='gray')
            title = f"GT: {ex['gt']} | Pred: {ex['pred']} | CER: {ex['cer']:.2f}"
            ax.set_title(title, fontsize=10, fontname='sans-serif')
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
    except Exception as e:
        print(f"Warning: Matplotlib contact sheet generation failed (likely font issues): {e}")

def select_representative_examples(predictions):
    correct = [p for p in predictions if p['cer'] == 0.0]
    partial = [p for p in predictions if 0.0 < p['cer'] <= 0.5]
    severe = [p for p in predictions if p['cer'] > 0.5 and p['pred'].strip() != ""]
    empty = [p for p in predictions if p['pred'].strip() == ""]
    
    selected = []
    selected.extend(correct[:4])
    selected.extend(partial[:4])
    selected.extend(severe[:4])
    selected.extend(empty[:4])
    
    # Fill remaining to make 10 if we don't have enough in categories
    if len(selected) < 10:
        for p in predictions:
            if p not in selected:
                selected.append(p)
            if len(selected) >= 10:
                break
                
    return selected

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CRNN model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/recognition/tamil/global_shuffle_lr_1e-4/best.pth", help="Path to checkpoint")
    parser.add_argument("--language", type=str, default="tamil", help="Language to evaluate on")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for validation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation/tamil/global_shuffle_lr_1e-4/", help="Directory to save evaluation results")
    args = parser.parse_args()
    
    print("============================================================")
    print("CRNN EVALUATION")
    print("============================================================")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output Directory: {args.output_dir}")
    
    config = load_config()
    working_root = config["dataset"]["working_root"]
    device = get_device()
    
    print(f"Device: {device}")
    
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
        
    vocab_path = os.path.join(working_root, "vocabulary", f"{args.language}_vocab.json")
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    print("Initializing CRNN architecture...")
    model = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Verify and load checkpoint
    print("Loading checkpoint weights...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        sys.exit(1)
        
    # Prepare Output Directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-load validation dataset
    print("Loading validation dataset...")
    val_loader = get_dataloader(
        dataset_root=working_root, 
        language=args.language, 
        split="validation", 
        tokenizer=tokenizer, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        max_packets=None, 
        target_packet=None,
        replay_previous=False,
        shuffle=False
    )
    
    if val_loader is None or len(val_loader) == 0:
        print("ERROR: Validation dataloader is empty.")
        sys.exit(1)
        
    print(f"Validation batches to process: {len(val_loader)}")
    
    # ----------------------------------------------------------------------
    # EVALUATION LOOP (Strictly NO TRAINING)
    # ----------------------------------------------------------------------
    model.eval()
    val_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    num_samples = 0
    
    all_predictions = []
    all_predicted_chars = []
    unique_words = set()
    total_raw_preds = 0
    blank_preds = 0
    exact_matches = 0
    
    print("Running evaluation...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            if len(batch_data) == 5:
                images, targets, target_lengths, actual_widths, image_paths = batch_data
            else:
                images, targets, target_lengths, actual_widths = batch_data
                image_paths = ["" for _ in range(images.size(0))]
            
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            actual_widths = actual_widths.to(device, non_blocking=True)
            
            outputs = model(images)
            outputs_perm = outputs.permute(1, 0, 2)
            outputs_log_probs = outputs_perm.log_softmax(2)
            
            input_lengths = (actual_widths // 4).to(torch.long)
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
                    
                for c in hyp_text:
                    all_predicted_chars.append(c)
                unique_words.add(hyp_text)
                    
                cer = calculate_cer(ref_text, hyp_text)
                wer = calculate_wer(ref_text, hyp_text)
                
                total_cer += cer
                total_wer += wer
                
                is_exact = (ref_text == hyp_text)
                if is_exact:
                    exact_matches += 1
                
                all_predictions.append({
                    "image_path": str(image_paths[i]),
                    "gt": ref_text,
                    "pred": hyp_text,
                    "cer": cer,
                    "exact_match": is_exact
                })
                    
                num_samples += 1
                
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                print(f"Processed batch {batch_idx + 1}/{len(val_loader)}")

    eval_time = time.time() - start_time
    
    # ----------------------------------------------------------------------
    # CALCULATE FINAL METRICS
    # ----------------------------------------------------------------------
    avg_loss = val_loss / len(val_loader)
    avg_cer = total_cer / num_samples if num_samples > 0 else 0
    avg_wer = total_wer / num_samples if num_samples > 0 else 0
    char_acc = max(0.0, 1.0 - avg_cer)
    word_acc = max(0.0, 1.0 - avg_wer)
    
    char_counts = Counter(all_predicted_chars)
    total_chars_predicted = sum(char_counts.values())
    
    blank_percentage = (blank_preds / total_raw_preds * 100) if total_raw_preds > 0 else 100.0
    avg_pred_length = (total_chars_predicted / num_samples) if num_samples > 0 else 0.0
    
    metrics = {
        "validation_loss": avg_loss,
        "CER": avg_cer,
        "WER": avg_wer,
        "character_accuracy": char_acc,
        "word_accuracy": word_acc,
        "exact_match_accuracy": exact_matches / num_samples if num_samples > 0 else 0,
        "blank_percentage": blank_percentage,
        "unique_predicted_characters": len(char_counts),
        "unique_predicted_sequences": len(unique_words),
        "average_prediction_length": avg_pred_length,
        "total_samples_evaluated": num_samples,
        "evaluation_time_seconds": eval_time
    }
    
    # ----------------------------------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------------------------------
    # 1. evaluation_results.json
    results_file = out_dir / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
        
    # 2. validation_predictions.json (limit to all or at least 100)
    # The prompt says "save at least 100 validation examples" - we will save all of them.
    predictions_file = out_dir / "validation_predictions.json"
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=4, ensure_ascii=False)
        
    # 3. evaluation_report.md
    report_file = out_dir / "evaluation_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# CRNN Evaluation Report\n\n")
        f.write(f"**Checkpoint:** `{args.checkpoint}`\n")
        f.write(f"**Total Samples:** {num_samples}\n\n")
        f.write("## Metrics\n")
        f.write(f"- Validation Loss: {avg_loss:.4f}\n")
        f.write(f"- CER: {avg_cer:.4f}\n")
        f.write(f"- WER: {avg_wer:.4f}\n")
        f.write(f"- Character Accuracy: {char_acc:.4f}\n")
        f.write(f"- Word/Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}\n")
        f.write(f"- Blank Prediction %: {blank_percentage:.2f}%\n")
        f.write(f"- Unique Predicted Characters: {len(char_counts)}\n")
        f.write(f"- Unique Predicted Sequences: {len(unique_words)}\n")
        f.write(f"- Average Prediction Length: {avg_pred_length:.2f}\n")
        
    # 4. Contact Sheet
    contact_sheet_examples = select_representative_examples(all_predictions)
    generate_contact_sheet(contact_sheet_examples, out_dir / "contact_sheet.png")
    
    # ----------------------------------------------------------------------
    # FINAL SUMMARY TERMINAL OUTPUT
    # ----------------------------------------------------------------------
    print("\n============================================================")
    print("EVALUATION SUMMARY")
    print("============================================================")
    print(f"Validation Loss:         {avg_loss:.4f}")
    print(f"CER:                     {avg_cer:.4f}")
    print(f"WER:                     {avg_wer:.4f}")
    print(f"Character Accuracy:      {char_acc:.4f}")
    print(f"Word Accuracy:           {word_acc:.4f}")
    print(f"Exact Match Accuracy:    {metrics['exact_match_accuracy']:.4f}")
    print(f"Blank Percentage:        {blank_percentage:.2f}%")
    print(f"Unique Chars Predicted:  {len(char_counts)}")
    print(f"Unique Seqs Predicted:   {len(unique_words)}")
    print(f"Average Pred Length:     {avg_pred_length:.2f}")
    print("------------------------------------------------------------")
    print(f"Results saved to: {out_dir.absolute()}")
    print(" - evaluation_results.json")
    print(" - validation_predictions.json")
    print(" - evaluation_report.md")
    print(" - contact_sheet.html / .png")
    print("============================================================\n")

if __name__ == "__main__":
    main()
