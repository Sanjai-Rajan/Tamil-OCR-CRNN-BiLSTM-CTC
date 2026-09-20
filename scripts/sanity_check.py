import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.digitalization.tokenizer import Tokenizer
from models.recognition.crnn import CRNN

def run_sanity_check():
    working_root = Path("data/tamil_ocr_dataset")
    vocab_path = working_root / "vocabulary" / "tamil_vocab.json"
    manifest_path = working_root / "imported" / "tamil" / "train" / "packet_001" / "manifest.jsonl"
    
    print("============================================================")
    print("SANITY CHECK: TOKENIZER & PIPELINE")
    print("============================================================")
    
    # 1. Tokenizer setup
    tokenizer = Tokenizer(vocab_path=str(vocab_path))
    print(f"Loaded Tamil Vocabulary. Size: {tokenizer.num_classes} classes (including blank)")
    
    # 2. Check 20 samples from dataset
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 20: break
            entry = json.loads(line)
            samples.append(entry)
            
    print("\n--- 20 Sample Label Validations ---")
    valid_decodes = 0
    empty_labels = 0
    for i, sample in enumerate(samples):
        label = sample.get("label", "")
        if not label:
            empty_labels += 1
            
        encoded = tokenizer.encode(label)
        decoded = tokenizer.decode(encoded)
        
        print(f"\nSample {i+1}:")
        print(f"Ground Truth: '{label}'")
        print(f"Encoded IDs:  {encoded}")
        print(f"Decoded:      '{decoded}'")
        
        if decoded == label:
            valid_decodes += 1
            
    print(f"\nValidation Summary: {valid_decodes}/20 matches. Empty labels: {empty_labels}")
    
    # 3. Model & CTC Sanity Check
    print("\n============================================================")
    print("SANITY CHECK: CTC LOSS & MODEL DIMENSIONS")
    print("============================================================")
    
    device = torch.device('cpu')
    model = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    model.eval()
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Load 1 real image
    img_path = working_root / "imported" / "tamil" / "train" / "packet_001" / samples[0]["image"]
    image = Image.open(img_path).convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image_tensor = transform(image).unsqueeze(0) # [B, C, H, W]
    
    label = samples[0]["label"]
    encoded = tokenizer.encode(label)
    targets = torch.tensor(encoded, dtype=torch.long)
    target_lengths = torch.tensor([len(encoded)], dtype=torch.long)
    
    print(f"Input image shape: {list(image_tensor.shape)}")
    
    with torch.no_grad():
        logits = model(image_tensor) # [B, T, C]
        print(f"CRNN output/logits shape: {list(logits.shape)}")
        
        logits_perm = logits.permute(1, 0, 2) # [T, B, C]
        print(f"Permuted shape: {list(logits_perm.shape)}")
        
        log_probs = logits_perm.log_softmax(2)
        print(f"After log_softmax: {list(log_probs.shape)}")
        
        input_lengths = torch.full(size=(log_probs.size(1),), fill_value=log_probs.size(0), dtype=torch.long)
        
        print(f"Target shape: {list(targets.shape)}")
        print(f"Target lengths: {target_lengths.tolist()}")
        print(f"Input lengths: {input_lengths.tolist()}")
        print(f"Vocabulary/classes: {tokenizer.num_classes}")
        
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        
        print(f"\nRaw logits min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")
        print(f"Log probability min/max: {log_probs.min().item():.4f} / {log_probs.max().item():.4f}")
        print(f"CTC Loss: {loss.item():.4f}")
        
        assert loss.item() >= 0, "Loss is negative! Malformed CTC input."
        assert not torch.isnan(loss), "Loss is NaN!"
        assert not torch.isinf(loss), "Loss is Inf!"
        
    print("\nAll sanity checks passed successfully!")

if __name__ == "__main__":
    run_sanity_check()
