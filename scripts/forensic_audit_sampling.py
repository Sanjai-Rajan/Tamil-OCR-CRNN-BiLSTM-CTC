import sys
import os
import json
import torch
from pathlib import Path
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode
from models.digitalization.tokenizer import Tokenizer
from scripts.dataset_loader import PacketOCRDataset

class AspectRatioPreservingResize:
    def __init__(self, height=32):
        self.height = height
        
    def __call__(self, img):
        w, h = img.size
        new_w = max(4, round(w * self.height / h))
        return img.resize((new_w, self.height), Image.Resampling.LANCZOS)

def ctc_min_length(target_text):
    length = len(target_text)
    for i in range(1, len(target_text)):
        if target_text[i] == target_text[i-1]:
            length += 1
    return length

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_path = "data/tamil_ocr_dataset/vocabulary/tamil_vocab.json"
    checkpoint_path = "checkpoints/recognition/tamil/ocr_balanced_temporal/best.pth"
    
    tokenizer = Tokenizer(vocab_path=vocab_path)
    model = CRNN(tokenizer.num_classes, feature_size=2048, less_downsample=True).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    dataset = PacketOCRDataset(
        dataset_root="data/tamil_ocr_dataset",
        language="tamil",
        split="test",
        tokenizer=tokenizer,
        transform=None
    )
    
    resize_op = AspectRatioPreservingResize(32)
    import torchvision.transforms as T
    to_tensor = T.ToTensor()
    normalize = T.Normalize((0.5,), (0.5,))
    
    buckets = {"1-5": [], "6-10": [], "11-15": [], "16+": []}
    
    for i in range(len(dataset)):
        item = dataset.data[i]
        img_path = item["image_path"]
        target = item["label"]
        target_len = len(target)
        
        b = "1-5" if target_len <= 5 else "6-10" if target_len <= 10 else "11-15" if target_len <= 15 else "16+"
        if len(buckets[b]) >= 5:
            continue
            
        try:
            img = Image.open(img_path).convert('L')
        except Exception:
            continue
            
        orig_w, orig_h = img.size
        resized_img = resize_op(img)
        res_w, res_h = resized_img.size
        
        # In actual batching there's padding, but for batch_size=1 it's just res_w
        pad_w = res_w
        
        tensor_img = normalize(to_tensor(resized_img)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor_img)
            decoded = ctc_decode(outputs)
            
        # outputs shape: [batch, seq_len, num_classes]
        temporal_len = outputs.size(1)
        
        # CTC deduplication and blank removal
        pred_seq = decoded[0].cpu().tolist()
        clean_pred = []
        prev = -1
        for p in pred_seq:
            if p != 0 and p != prev:
                clean_pred.append(p)
            prev = p
            
        pred_text = tokenizer.decode(clean_pred)
        pred_len = len(pred_text)
        
        min_ctc = ctc_min_length(target)
        truncated = (temporal_len < min_ctc)
        
        buckets[b].append({
            "target": target,
            "prediction": pred_text,
            "target_len": target_len,
            "pred_len": pred_len,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "aspect_ratio": orig_w / orig_h,
            "model_input_width": pad_w,
            "temporal_length": temporal_len,
            "truncated": truncated,
            "min_ctc_len": min_ctc
        })
        
        if all(len(v) >= 5 for v in buckets.values()):
            break
            
    with open("docs/robustness_samples.json", "w", encoding="utf-8") as f:
        json.dump(buckets, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
