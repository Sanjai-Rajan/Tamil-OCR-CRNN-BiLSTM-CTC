import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    print("=== MLM SMOKE TEST ===")
    model_dir = PROJECT_ROOT / "models" / "indic_bert"
    
    print(f"Loading IndicBERT from {model_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        model = AutoModelForMaskedLM.from_pretrained(str(model_dir), local_files_only=True)
        print("PASS: Tokenizer and Model loaded successfully.")
    except Exception as e:
        print(f"FAIL: {e}")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    text = "இது ஒரு தமிழ் [MASK] ஆகும்."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Fake label for a quick backward pass
    labels = inputs.input_ids.clone()
    
    model.train()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    
    if loss is not None and not torch.isnan(loss):
        print("PASS: Forward pass successful.")
    else:
        print("FAIL: Forward pass failed.")
        
    loss.backward()
    
    has_grad = any(p.grad is not None for p in model.parameters())
    if has_grad:
        print("PASS: Backward pass successful.")
    else:
        print("FAIL: Backward pass failed.")
        
    print("ALL TESTS PASSED.")

if __name__ == "__main__":
    main()
