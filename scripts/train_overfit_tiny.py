import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path

# Force UTF-8 encoding for standard output on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.recognition.crnn import CRNN
from models.recognition.decoder import ctc_decode
from scripts.dataset_loader import get_dataloader
from models.digitalization.tokenizer import Tokenizer
from utils.device import get_device
from utils.metrics import calculate_cer, calculate_wer

def train_tiny():
    working_root = "data/tamil_ocr_dataset"
    device = get_device()
    vocab_path = os.path.join(working_root, "vocabulary", "tamil_vocab.json")
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    # Force loading just a single packet
    print("Loading tiny dataset...")
    dataloader = get_dataloader(
        dataset_root=working_root, 
        language="tamil", 
        split="train", 
        tokenizer=tokenizer, 
        batch_size=32, 
        max_packets=1, 
        shuffle=False
    )
    
    if dataloader is None or len(dataloader.dataset) == 0:
        print("ERROR: Dataloader is empty.")
        sys.exit(1)
        
    # Manually extract just the first 32 samples (1 batch)
    tiny_dataset = []
    for i in range(min(32, len(dataloader.dataset))):
        tiny_dataset.append(dataloader.dataset[i])
        
    from torch.utils.data import DataLoader
    from scripts.dataset_loader import crnn_collate_fn
    tiny_loader = DataLoader(
        tiny_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=crnn_collate_fn
    )

    model = CRNN(tokenizer.num_classes, feature_size=2048, less_downsample=True).to(device)
    # Using a higher learning rate to overfit quickly
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    print(f"Starting Tiny Overfit Test on {len(tiny_dataset)} samples...")
    
    for epoch in range(1, 501):
        model.train()
        
        for batch_data in tiny_loader:
            if len(batch_data) == 5:
                images, targets, target_lengths, actual_widths, _ = batch_data
            else:
                images, targets, target_lengths, actual_widths = batch_data
                
            images = images.to(device)
            targets = targets.to(device)
            actual_widths = actual_widths.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)
            outputs_log_probs = outputs.float().log_softmax(2)
            
            input_lengths = torch.round((actual_widths.float() / images.size(3)) * outputs.size(0)).to(torch.long)
            input_lengths = torch.clamp(input_lengths, min=1, max=outputs.size(0))
            
            loss = criterion(outputs_log_probs, targets, input_lengths, target_lengths)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
        if epoch % 50 == 0 or epoch == 1:
            # Eval on same tiny dataset
            model.eval()
            total_cer = 0
            with torch.no_grad():
                preds = ctc_decode(outputs.permute(1, 0, 2)) # [B, T]
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
                        
                    cer = calculate_cer(ref_text, hyp_text)
                    total_cer += cer
                    
                    if i == 0 and epoch % 100 == 0:
                        print(f"  [Sample 1] GT: '{ref_text}' | PRED: '{hyp_text}'")
            
            avg_cer = total_cer / len(target_lengths)
            print(f"Epoch {epoch}/500 | Loss: {loss.item():.4f} | CER: {avg_cer:.4f}")
            if avg_cer < 0.05:
                print("Tiny overfit test successful!")
                break
                
if __name__ == "__main__":
    train_tiny()
