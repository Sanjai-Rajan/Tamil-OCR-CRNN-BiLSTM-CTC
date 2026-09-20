import sys
import os
import json
import time
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import Levenshtein
from torch.utils.data import Dataset, DataLoader

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from models.recognition.crnn import CRNN
from models.digitalization.tokenizer import Tokenizer
from utils.metrics import calculate_cer, calculate_wer

class AspectRatioPreservingResize:
    def __init__(self, height=32):
        self.height = height
        
    def __call__(self, img):
        w, h = img.size
        new_w = max(4, round(w * self.height / h))
        return img.resize((new_w, self.height), Image.Resampling.LANCZOS)

class PacketOCRDataset(Dataset):
    def __init__(self, test_dir, transform):
        self.data = []
        self.transform = transform
        for packet in sorted(Path(test_dir).iterdir()):
            if not packet.is_dir() or not packet.name.startswith("packet_"): continue
            manifest = packet / "manifest.jsonl"
            if manifest.exists():
                with open(manifest, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            self.data.append({
                                "image_path": str(packet / entry["image"]),
                                "label": entry["label"]
                            })
                            
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item["image_path"]).convert('L')
        except:
            image = Image.new('L', (100, 32), color=255)
        if self.transform:
            image = self.transform(image)
        return image, item["label"], item["image_path"], idx

def crnn_collate_fn(batch):
    images, labels, paths, idxs = zip(*batch)
    actual_widths = [img.size(2) for img in images]
    max_width = max(actual_widths)
    height = images[0].size(1)
    
    padded_images = torch.ones(len(images), 1, height, max_width) * 1.0
    for i, img in enumerate(images):
        padded_images[i, :, :, :img.size(2)] = img
        
    return padded_images, labels, paths, idxs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device, flush=True)
    
    ckpt_path = root_dir / "checkpoints" / "recognition" / "tamil" / "fast_track_ctc_fix" / "epoch_011.pth"
    if not ckpt_path.exists(): ckpt_path = root_dir / "checkpoints" / "recognition" / "tamil" / "fast_track_ctc_fix" / "best.pth"
    if not ckpt_path.exists(): ckpt_path = root_dir / "checkpoints" / "recognition" / "tamil" / "fast_track_ctc_fix" / "latest.pth"
        
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    w = state_dict.get('sequence.lstm.weight_ih_l0')
    feature_size = w.shape[1] if w is not None else 2048
    print(f"Inferred feature size: {feature_size}", flush=True)
    
    vocab_path = root_dir / "data" / "tamil_ocr_dataset" / "vocabulary" / "tamil_vocab.json"
    tokenizer = Tokenizer(vocab_path=str(vocab_path))
    
    model = CRNN(tokenizer.num_classes, feature_size=feature_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    transform = transforms.Compose([
        AspectRatioPreservingResize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dir = root_dir / "data" / "tamil_ocr_dataset" / "imported" / "tamil" / "test"
    dataset = PacketOCRDataset(test_dir, transform)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4, collate_fn=crnn_collate_fn)
    
    print(f"Loaded {len(dataset)} test samples. Evaluating in batches...")
    
    total_cer, total_wer, char_acc_sum, word_acc_sum = 0.0, 0.0, 0.0, 0.0
    total_blanks, total_preds, total_pred_len = 0, 0, 0
    predictions = []
    
    with torch.no_grad():
        for batch_images, labels, paths, idxs in dataloader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            
            if outputs.dim() == 3:
                probs = torch.softmax(outputs, dim=-1)
                
                # Assume shape [T, B, C] for standard CRNN or [B, T, C]
                if outputs.size(0) == batch_images.size(0):
                    # [B, T, C]
                    seqs = outputs.argmax(-1)
                    confs = probs.max(-1).values
                else:
                    # [T, B, C] -> permute to [B, T, C]
                    seqs = outputs.permute(1, 0, 2).argmax(-1)
                    confs = probs.permute(1, 0, 2).max(-1).values
                    
                seqs = seqs.cpu().tolist()
                confs = confs.cpu().tolist()
                
                for b_idx in range(len(labels)):
                    seq = seqs[b_idx]
                    prob = confs[b_idx]
                    label = labels[b_idx]
                    
                    clean_pred, clean_prob = [], []
                    prev = -1
                    blanks_in_seq = 0
                    for p, pr in zip(seq, prob):
                        if p == 0: blanks_in_seq += 1
                        if p != 0 and p != prev:
                            clean_pred.append(p)
                            clean_prob.append(pr)
                        prev = p
                        
                    total_blanks += blanks_in_seq
                    total_preds += len(seq)
                    
                    pred_text = tokenizer.decode(clean_pred)
                    total_pred_len += len(pred_text)
                    
                    c = calculate_cer(label, pred_text)
                    w = calculate_wer(label, pred_text)
                    ca = max(0.0, 1.0 - c)
                    wa = max(0.0, 1.0 - w)
                    
                    total_cer += c
                    total_wer += w
                    char_acc_sum += ca
                    word_acc_sum += wa
                    
                    if len(predictions) < 20000: # store only to a limit or just store all
                        predictions.append({
                            "sample": idxs[b_idx],
                            "gt": label,
                            "pred": pred_text,
                            "pred_len": len(pred_text),
                            "conf": sum(clean_prob)/len(clean_prob) if clean_prob else 0.0,
                            "blanks": blanks_in_seq,
                            "seq_len": len(seq)
                        })
            
            print(f"Processed {len(predictions)}/{len(dataset)}...", flush=True)
            
    n = len(dataset)
    print(f"Processed {n} samples")
    print(f"CER: {total_cer/n:.4f}")
    print(f"WER: {total_wer/n:.4f}")
    print(f"Char Acc: {char_acc_sum/n:.4f}")
    print(f"Word Acc: {word_acc_sum/n:.4f}")
    print(f"Blank %: {total_blanks/total_preds:.4f}")
    print(f"Avg Pred Len: {total_pred_len/n:.4f}")
    
    out_dir = root_dir / "outputs" / "fast_track"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "epoch11_predictions.txt", "w", encoding="utf-8") as f:
        for p in predictions[:30]:
            f.write(f"Sample #{p['sample']}\n")
            f.write(f"Ground Truth: {p['gt']}\n")
            f.write(f"Prediction: {p['pred']}\n")
            f.write(f"Prediction Length: {p['pred_len']}\n")
            f.write(f"Confidence: {p['conf']:.4f}\n")
            f.write("-" * 20 + "\n")
            
    err_A, err_B, err_C, err_D, err_E, err_F, err_G, err_H = 0, 0, 0, 0, 0, 0, 0, 0
    for p in predictions:
        if p["gt"] == p["pred"]: continue
        gt = p["gt"]
        pr = p["pred"]
        if len(pr) == 0: err_A += 1
        elif len(pr) < len(gt) and gt.startswith(pr): err_B += 1
        elif len(pr) == len(gt): err_C += 1
        elif len(pr) < len(gt): err_D += 1
        elif len(pr) > len(gt): err_E += 1
        else: err_H += 1
            
    print("--- ERROR ANALYSIS ---")
    print(f"A. Blank collapse: {err_A}")
    print(f"B. Right-side truncation: {err_B}")
    print(f"C. Character substitution: {err_C}")
    print(f"D. Character deletion: {err_D}")
    print(f"E. Character insertion: {err_E}")
    print(f"F. Tamil vowel/diacritic confusion: Not easily detected by script, check manually")
    print(f"G. Repeated-character collapse: Not easily detected, check manually")
    print(f"H. Other: {err_H}")

if __name__ == "__main__":
    main()
