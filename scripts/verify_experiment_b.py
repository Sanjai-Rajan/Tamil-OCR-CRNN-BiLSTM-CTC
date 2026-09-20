import torch
import torch.nn as nn
import time
from models.recognition.crnn import CRNN
from models.recognition.crnn_v2 import CRNNV2

def sum_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("=== PARAMETER COMPARISON ===")
    crnn_a = CRNN(num_classes=77, feature_size=2048, less_downsample=True)
    crnn_b = CRNNV2(num_classes=77, feature_size=4096, high_vertical_resolution=True)

    params_a = sum_params(crnn_a)
    params_b = sum_params(crnn_b)
    
    print(f"Experiment A (CRNN): {params_a:,} parameters")
    print(f"Experiment B (CRNNV2): {params_b:,} parameters")
    print(f"Difference: +{params_b - params_a:,} parameters")

    print("\n=== SHAPE VERIFICATION ===")
    batch_size = 4
    height = 32
    width = 200
    images = torch.randn(batch_size, 1, height, width)
    
    out_a = crnn_a(images)
    out_b = crnn_b(images)
    
    print(f"Input image shape: {images.shape}")
    print(f"Experiment A output shape (B, T, C): {out_a.shape}")
    print(f"Experiment B output shape (B, T, C): {out_b.shape}")
    
    if out_a.shape[1] == out_b.shape[1]:
        print("-> SUCCESS: Horizontal sequence length (T) is identical.")
    else:
        print("-> ERROR: Sequence lengths differ!")

    print("\n=== CTC VERIFICATION ===")
    # Create fake targets
    targets = torch.randint(1, 77, (batch_size * 10,))
    target_lengths = torch.tensor([10] * batch_size)
    
    # Calculate input lengths dynamically as in Experiment A
    actual_widths = torch.tensor([width] * batch_size)
    out_perm = out_b.permute(1, 0, 2)
    input_lengths = torch.round((actual_widths.float() / images.size(3)) * out_perm.size(0)).to(torch.long)
    input_lengths = torch.clamp(input_lengths, max=out_perm.size(0))

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    log_probs = out_perm.log_softmax(2)
    
    print(f"log_probs shape: {log_probs.shape}")
    print(f"targets length: {targets.shape}")
    print(f"input_lengths: {input_lengths}")
    print(f"target_lengths: {target_lengths}")

    start = time.time()
    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    loss.backward()
    end = time.time()
    
    print(f"CTC Loss: {loss.item()}")
    print(f"Forward + Backward pass time: {end - start:.4f}s")
    
    if torch.isfinite(loss):
        print("-> SUCCESS: Loss is finite. No NaN. Sequence/Backward pass successful.")
    else:
        print("-> ERROR: Loss is infinite or NaN.")

if __name__ == "__main__":
    main()
