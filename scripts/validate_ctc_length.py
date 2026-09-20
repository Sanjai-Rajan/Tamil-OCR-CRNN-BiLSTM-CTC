import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from models.recognition.crnn import CRNN

def main():
    model = CRNN(num_classes=77, feature_size=2048, less_downsample=True)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Simulate a batch with different widths
    # Max width 200, one is 150
    images = torch.randn(2, 1, 32, 200)
    actual_widths = torch.tensor([200, 150])
    
    outputs = model(images)
    outputs_perm = outputs.permute(1, 0, 2)
    outputs_log_probs = outputs_perm.log_softmax(2)
    seq_len = outputs_perm.size(0)
    
    print("=== CTC LENGTH VALIDATION ===")
    print(f"Input width (padded max): {images.size(3)}")
    print(f"CRNN output sequence length: {seq_len}")
    
    # Target sequences
    targets = torch.tensor([10, 20, 30, 40, 50, 11, 21, 31], dtype=torch.long)
    target_lengths = torch.tensor([5, 3], dtype=torch.long)
    print(f"Target lengths: {target_lengths.tolist()}")
    
    # New dynamic calculation
    new_lengths = torch.round((actual_widths.float() / images.size(3)) * seq_len).to(torch.long)
    new_lengths = torch.clamp(new_lengths, min=1, max=seq_len)
    print(f"CTC input length: {new_lengths.tolist()}")
    
    loss = criterion(outputs_log_probs, targets, new_lengths, target_lengths)
    loss.backward()
    
    if torch.isnan(loss) or torch.isinf(loss):
        print("CTC loss: INVALID")
    else:
        print(f"CTC loss: FINITE ({loss.item():.4f})")

if __name__ == "__main__":
    main()
