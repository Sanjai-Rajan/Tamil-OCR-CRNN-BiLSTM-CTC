import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from scripts.dataset_loader import get_dataloader
from models.digitalization.tokenizer import Tokenizer

def main():
    print("Initializing tokenizer...")
    vocab_path = "data/tamil_ocr_dataset/vocabulary/tamil_vocab.json"
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    print("Loading dataloader with multi-scale augmentation...")
    train_loader = get_dataloader(
        dataset_root="data/tamil_ocr_dataset",
        language="tamil",
        split="train",
        tokenizer=tokenizer,
        batch_size=8,
        num_workers=0,
        max_packets=1,
        shuffle=True,
        use_multiscale_aug=True
    )
    
    out_dir = Path("outputs/training/tamil/refinement/exp_A_multiscale/previews")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Grab one batch
    for images, targets, target_lengths, widths, img_paths in train_loader:
        print(f"Batch images shape: {images.shape}")
        
        for i in range(len(images)):
            # Un-normalize
            tensor = (images[i] * 0.5) + 0.5
            pil_img = transforms.ToPILImage()(tensor)
            
            # The tensor is padded to max_width in the batch, so there will be black padding on the right.
            # But the content should be correctly scaled.
            
            # Save
            path = out_dir / f"preview_{i}.png"
            pil_img.save(path)
            print(f"Saved {path}")
            
            # Print the target text for sanity
            start_idx = sum(target_lengths[:i]) if i > 0 else 0
            target_seq = targets[start_idx:start_idx+target_lengths[i]].tolist()
            text = tokenizer.decode(target_seq)
            print(f"  Target: {text}")
            
        break
        
    print("Preview generation complete.")

if __name__ == "__main__":
    main()
