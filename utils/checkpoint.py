import torch
import os

def save_model(model, path="checkpoints/model.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path="checkpoints/model.pth"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    return model

def save_optimizer(optimizer, path="checkpoints/optimizer.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(optimizer.state_dict(), path)

def load_optimizer(optimizer, path="checkpoints/optimizer.pth"):
    if os.path.exists(path):
        optimizer.load_state_dict(torch.load(path))
    return optimizer
