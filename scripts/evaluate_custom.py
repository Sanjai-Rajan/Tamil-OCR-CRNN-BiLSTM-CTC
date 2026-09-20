import sys
import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
from pathlib import Path
import time
from datetime import datetime
from collections import Counter
from torchvision import transforms
from PIL import Image

# Force UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Ensure the project root is in the path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode
from scripts.dataset_loader import AspectRatioPreservingResize
from models.digitalization.tokenizer import Tokenizer
from utils.device import get_device
from utils.metrics import calculate_cer, calculate_wer
from utils.logger import get_logger

# Simple logging for CLI
logger = get_logger("evaluate_custom")

def load_config(config_path="config/model_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def generate_alignment(ref_text, hyp_text):
    """
    Generates a simple visual alignment string for differences.
    (Simple version: prints char by char if lengths differ heavily, it's just for visualization)
    """
    import difflib
    sm = difflib.SequenceMatcher(None, ref_text, hyp_text)
    
    gt_viz = ""
    pred_viz = ""
    err_viz = ""
    
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            gt_viz += ref_text[i1:i2]
            pred_viz += hyp_text[j1:j2]
            err_viz += " " * (i2 - i1)
        elif tag == 'replace':
            gt_viz += ref_text[i1:i2]
            pred_viz += hyp_text[j1:j2]
            err_viz += "^" * max(i2 - i1, j2 - j1)
            # pad shorter string with spaces for alignment
            diff = abs((i2 - i1) - (j2 - j1))
            if (i2 - i1) > (j2 - j1):
                pred_viz += " " * diff
            else:
                gt_viz += " " * diff
        elif tag == 'delete':
            gt_viz += ref_text[i1:i2]
            pred_viz += "-" * (i2 - i1)
            err_viz += "D" * (i2 - i1)
        elif tag == 'insert':
            gt_viz += "-" * (j2 - j1)
            pred_viz += hyp_text[j1:j2]
            err_viz += "I" * (j2 - j1)
            
    return gt_viz, pred_viz, err_viz

def evaluate_images(image_paths, ground_truths, args):
    device = get_device()
    logger.info(f"Using device: {device}")
    
    config = load_config(os.path.join(project_root, "config", "model_config.yaml"))
    working_root = config["dataset"]["working_root"]
    
    vocab_path = os.path.join(project_root, working_root, "vocabulary", f"{args.language}_vocab.json")
    if not os.path.exists(vocab_path):
        # Fallback to absolute if it was defined absolutely in config
        vocab_path = os.path.join(working_root, "vocabulary", f"{args.language}_vocab.json")
        if not os.path.exists(vocab_path):
            logger.error(f"Vocabulary not found at {vocab_path}")
            sys.exit(1)
            
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    logger.info("Initializing CRNN architecture...")
    model = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    
    checkpoint_path = os.path.join(project_root, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        # fallback to direct path
        if os.path.exists(args.checkpoint):
            checkpoint_path = args.checkpoint
        else:
            logger.error(f"Checkpoint not found at {args.checkpoint}")
            sys.exit(1)
            
    logger.info("Loading checkpoint weights...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Fallback if checkpoint is just the state dict
            model.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded successfully.")
    except Exception as e:
        logger.error(f"Failed loading checkpoint: {e}")
        sys.exit(1)
        
    model.eval()
    
    # Using exact same preprocessing as dataset_loader.py
    transform = transforms.Compose([
        AspectRatioPreservingResize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    results = []
    total_cer = 0.0
    total_wer = 0.0
    exact_matches = 0
    evaluated_with_gt = 0
    
    logger.info("Starting inference...")
    
    for i, img_path in enumerate(image_paths):
        gt = ground_truths[i] if i < len(ground_truths) else None
        logger.info(f"Processing ({i+1}/{len(image_paths)}): {img_path}")
        
        try:
            image = Image.open(img_path).convert('L')
            input_tensor = transform(image).unsqueeze(0).to(device)
        except Exception as e:
            logger.error(f"Failed to process image {img_path}: {e}")
            continue
            
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
                
        result = {
            "image": os.path.basename(img_path),
            "image_path": img_path,
            "prediction": pred_text
        }
        
        print(f"\nImage: {os.path.basename(img_path)}")
        
        if gt is not None:
            cer = calculate_cer(gt, pred_text)
            wer = calculate_wer(gt, pred_text)
            
            total_cer += cer
            total_wer += wer
            evaluated_with_gt += 1
            is_exact = (gt == pred_text)
            if is_exact:
                exact_matches += 1
                
            result["ground_truth"] = gt
            result["cer"] = cer
            result["wer"] = wer
            result["character_accuracy"] = max(0.0, 1.0 - cer)
            result["word_accuracy"] = max(0.0, 1.0 - wer)
            result["exact_match"] = is_exact
            
            print(f"Ground truth: {gt}")
            print(f"Prediction:   {pred_text}")
            
            if not is_exact:
                gt_viz, pred_viz, err_viz = generate_alignment(gt, pred_text)
                print(f"Alignment:")
                print(f"GT:   {gt_viz}")
                print(f"Pred: {pred_viz}")
                print(f"Diff: {err_viz}")
                
            print(f"CER: {cer*100:.2f}% | WER: {wer*100:.2f}% | Char Acc: {result['character_accuracy']*100:.2f}% | Word Acc: {result['word_accuracy']*100:.2f}% | Exact Match: {'YES' if is_exact else 'NO'}")
        else:
            print(f"Prediction:   {pred_text}")
            
        results.append(result)
        
    # Aggregate Metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(project_root) / args.output_dir / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    aggregate_metrics = {
        "model_checkpoint": args.checkpoint,
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_images": len(results),
        "evaluated_with_ground_truth": evaluated_with_gt
    }
    
    if evaluated_with_gt > 0:
        avg_cer = total_cer / evaluated_with_gt
        avg_wer = total_wer / evaluated_with_gt
        
        aggregate_metrics.update({
            "average_cer": avg_cer,
            "average_wer": avg_wer,
            "average_character_accuracy": max(0.0, 1.0 - avg_cer),
            "average_word_accuracy": max(0.0, 1.0 - avg_wer),
            "exact_match_accuracy": exact_matches / evaluated_with_gt
        })
        
        print("\n========================================")
        print("AGGREGATE METRICS")
        print("========================================")
        print(f"Number of images: {evaluated_with_gt}")
        print(f"Average CER:      {avg_cer*100:.2f}%")
        print(f"Average WER:      {avg_wer*100:.2f}%")
        print(f"Char Accuracy:    {aggregate_metrics['average_character_accuracy']*100:.2f}%")
        print(f"Word Accuracy:    {aggregate_metrics['average_word_accuracy']*100:.2f}%")
        print(f"Exact Match Acc:  {aggregate_metrics['exact_match_accuracy']*100:.2f}%")
        print("========================================")
        
    output_data = {
        "metadata": aggregate_metrics,
        "results": results
    }
    
    output_file = out_dir / "results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
    print(f"\nDetailed results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Tamil OCR model on custom images")
    parser.add_argument("--image", type=str, help="Path to a single test image")
    parser.add_argument("--input-dir", type=str, help="Directory containing multiple test images")
    parser.add_argument("--ground-truth", type=str, help="Ground truth text for the single image")
    parser.add_argument("--gt-file", type=str, help="JSON file mapping filenames to ground truth texts")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/recognition/tamil/global_shuffle_lr_1e-4/best.pth", help="Path to checkpoint")
    parser.add_argument("--language", type=str, default="tamil", help="Language for vocabulary")
    parser.add_argument("--output-dir", type=str, default="outputs/custom_evaluation", help="Directory to save evaluation results")
    args = parser.parse_args()
    
    if not args.image and not args.input_dir:
        print("Error: Must provide either --image or --input-dir")
        parser.print_help()
        sys.exit(1)
        
    image_paths = []
    ground_truths = []
    
    # Handle GT mappings
    gt_mapping = {}
    if args.gt_file and os.path.exists(args.gt_file):
        with open(args.gt_file, "r", encoding="utf-8") as f:
            gt_mapping = json.load(f)
            
    # Single image case
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            sys.exit(1)
        image_paths.append(args.image)
        if args.ground_truth:
            ground_truths.append(args.ground_truth)
        elif os.path.basename(args.image) in gt_mapping:
            ground_truths.append(gt_mapping[os.path.basename(args.image)])
        else:
            ground_truths.append(None)
            
    # Directory case
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            print(f"Error: Directory not found at {args.input_dir}")
            sys.exit(1)
            
        valid_exts = {'.png', '.jpg', '.jpeg', '.webp'}
        for f in os.listdir(args.input_dir):
            if Path(f).suffix.lower() in valid_exts:
                img_path = os.path.join(args.input_dir, f)
                image_paths.append(img_path)
                
                if f in gt_mapping:
                    ground_truths.append(gt_mapping[f])
                else:
                    ground_truths.append(None)
                    
        if not image_paths:
            print(f"No valid images found in {args.input_dir}")
            sys.exit(1)
            
    evaluate_images(image_paths, ground_truths, args)

if __name__ == "__main__":
    main()
