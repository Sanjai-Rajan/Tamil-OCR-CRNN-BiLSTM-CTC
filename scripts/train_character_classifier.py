import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.recognition.character_classifier import CharacterClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")
    num_classes = 77
    
    model = CharacterClassifier(num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    if args.smoke_test:
        print("Running CPU smoke test...")
        dummy_input = torch.randn(4, 1, 64, 64).to(device)
        dummy_target = torch.randint(0, num_classes, (4,)).to(device)
        
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        
        print("Smoke test passed.")
        return

if __name__ == "__main__":
    main()
