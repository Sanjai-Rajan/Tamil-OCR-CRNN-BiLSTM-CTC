import sys
import os
import torch
import json
import time
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
sys.stdout.reconfigure(encoding='utf-8')

from models.recognition.crnn import CRNN
from app.inference import OCRService
from models.digitalization.tokenizer import Tokenizer
from models.recognition.decoder import ctc_decode_with_confidence

def get_config(ckpt_path):
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location='cpu')
    return ckpt

def analyze_model_architecture(ckpt):
    state_dict = ckpt['model_state_dict']
    shapes = {}
    for k, v in state_dict.items():
        if 'weight' in k or 'bias' in k:
            shapes[k] = list(v.shape)
    return shapes

def simple_greedy_decoder(outputs):
    # outputs: [batch, seq_len, num_classes]
    preds = torch.argmax(outputs, dim=2)
    return preds

def main():
    print("=== FINAL EPOCH-11 MODEL FORENSIC ANALYSIS ===\n")
    
    ckpt_a_path = root_dir / "checkpoints" / "recognition" / "tamil" / "fast_track_ctc_fix" / "best.pth"
    ckpt_b_path = root_dir / "checkpoints" / "recognition" / "tamil" / "tamil_full_40epoch" / "best.pth"
    
    if not ckpt_a_path.exists():
        print(f"Error: Experiment A checkpoint not found at {ckpt_a_path}")
        return
        
    print("1. CHECKPOINT CONFIGURATION & COMPATIBILITY")
    ckpt_a = get_config(ckpt_a_path)
    # Print some info if available
    print(f"Epoch A: {ckpt_a.get('epoch', 'N/A')}")
    print(f"Loss A: {ckpt_a.get('val_loss', 'N/A')}")
    
    print("\n2. COMPARE BASELINE AND EXPERIMENT A ARCHITECTURES")
    if ckpt_b_path.exists():
        ckpt_b = get_config(ckpt_b_path)
        print("Baseline checkpoint found. Comparing architectures...")
        shapes_a = analyze_model_architecture(ckpt_a)
        shapes_b = analyze_model_architecture(ckpt_b)
        diff = []
        for k in shapes_b.keys():
            if k not in shapes_a:
                diff.append(f"Missing in A: {k}")
            elif shapes_a[k] != shapes_b[k]:
                diff.append(f"Shape mismatch in {k}: A={shapes_a[k]}, Baseline={shapes_b[k]}")
        for k in shapes_a.keys():
            if k not in shapes_b:
                diff.append(f"Extra in A: {k}")
        
        if not diff:
            print("Architecture weights shapes are IDENTICAL.")
        else:
            for d in diff:
                print(d)
    else:
        print("Baseline checkpoint not found for direct comparison.")
        ckpt_b = None

    print("\n3. COMPARE TRAINING CONFIGURATION")
    # Will do this in the markdown manually if needed, or check json
    train_a_json = root_dir / "outputs" / "training" / "tamil" / "fast_track_ctc_fix" / "training_history.json"
    if train_a_json.exists():
        with open(train_a_json, 'r') as f:
            hist_a = json.load(f)
            print("Experiment A config sample:")
            if hist_a and isinstance(hist_a, list):
                print(json.dumps(hist_a[-1], indent=2))
            else:
                print(hist_a)
    else:
        print("Experiment A training history not found.")

    print("\n4. CHECK RAW CTC OUTPUT & 5. DECODER FORENSICS")
    # Load 5 images
    img_dir = root_dir / "data" / "tamil_ocr_dataset" / "imported" / "tamil" / "train" / "packet_014" / "images"
    test_images = list(img_dir.glob("*.jpg"))[:5]
    
    ocr_a = OCRService()
    ocr_a.checkpoint_path = ckpt_a_path
    ocr_a.load_model()
    
    ocr_b = None
    if ckpt_b:
        ocr_b = OCRService()
        ocr_b.checkpoint_path = ckpt_b_path
        ocr_b.load_model()
        
    print("\n=== SINGLE-LINE RAW CTC & BASELINE TEST ===")
    for i, img_p in enumerate(test_images):
        print(f"\n--- Image {i+1} : {img_p.name} ---")
        img = Image.open(img_p).convert('L')
        img_tensor = ocr_a.transform(img).unsqueeze(0).to(ocr_a.device)
        
        with torch.no_grad():
            outputs_a = ocr_a.model(img_tensor) # [1, seq_len, num_classes]
            
        probs_a = F.softmax(outputs_a, dim=-1) # [1, seq_len, num_classes]
        seq_len = probs_a.size(1)
        num_classes = probs_a.size(2)
        
        # Analyze blank vs non-blank (assuming blank=0)
        blank_probs = probs_a[0, :, 0]
        avg_blank = blank_probs.mean().item()
        
        non_blank_probs = probs_a[0, :, 1:]
        max_non_blank, _ = torch.max(non_blank_probs, dim=-1)
        avg_max_non_blank = max_non_blank.mean().item()
        
        high_conf = (max_non_blank > 0.5).sum().item()
        argmax_preds = simple_greedy_decoder(outputs_a)[0]
        blank_ratio = (argmax_preds == 0).float().mean().item()
        
        print("Experiment A Raw CTC Stats:")
        print(f"Sequence length: {seq_len}, Classes: {num_classes}")
        print(f"Avg Blank Prob: {avg_blank:.4f}")
        print(f"Avg Max Non-Blank Prob: {avg_max_non_blank:.4f}")
        print(f"High-conf non-blank timesteps (>0.5): {high_conf}/{seq_len}")
        print(f"Greedy Argmax Blank Ratio: {blank_ratio:.4f}")
        
        # Decoder 1: Greedy (Simple)
        pred_greedy = argmax_preds.cpu().tolist()
        clean_greedy = []
        prev = -1
        for p in pred_greedy:
            if p != 0 and p != prev:
                clean_greedy.append(p)
            prev = p
        greedy_text = ocr_a.tokenizer.decode(clean_greedy)
        print(f"Experiment A Greedy Decoder Output ({len(greedy_text)} chars): '{greedy_text}'")
        
        # Decoder 2: ctc_decode_with_confidence
        decoded_indices, confidences = ctc_decode_with_confidence(outputs_a)
        pred_seq = decoded_indices[0].cpu().tolist()
        clean_conf = []
        clean_pred = []
        prev = -1
        for p, conf in zip(pred_seq, confidences[0]):
            if p != 0 and p != prev:
                clean_pred.append(p)
                clean_conf.append(conf)
            prev = p
        conf_text = ocr_a.tokenizer.decode(clean_pred)
        avg_conf = sum(clean_conf)/len(clean_conf) if clean_conf else 0.0
        print(f"Experiment A CTC Decoder Output ({len(conf_text)} chars, conf={avg_conf:.4f}): '{conf_text}'")
        
        # Baseline Comparison
        if ocr_b:
            with torch.no_grad():
                outputs_b = ocr_b.model(img_tensor)
            decoded_b, confs_b = ctc_decode_with_confidence(outputs_b)
            pred_b = decoded_b[0].cpu().tolist()
            clean_b = []
            clean_conf_b = []
            prev_b = -1
            for p, conf in zip(pred_b, confs_b[0]):
                if p != 0 and p != prev_b:
                    clean_b.append(p)
                    clean_conf_b.append(conf)
                prev_b = p
            text_b = ocr_b.tokenizer.decode(clean_b)
            
            # baseline blank ratio
            preds_b = simple_greedy_decoder(outputs_b)[0]
            blank_ratio_b = (preds_b == 0).float().mean().item()
            avg_conf_b = sum(clean_conf_b)/len(clean_conf_b) if clean_conf_b else 0.0
            
            print(f"Baseline Decoder Output ({len(text_b)} chars, conf={avg_conf_b:.4f}): '{text_b}'")
            print(f"Baseline Greedy Blank Ratio: {blank_ratio_b:.4f}")


if __name__ == "__main__":
    main()
