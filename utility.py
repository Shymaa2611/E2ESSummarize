import os 
import torch
import torch.nn as nn


def save_model(model, optimizer, folder_path='checkpoint', filename='checkpoint.pt'):
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(model, optimizer, folder_path='checkpoint', filename='checkpoint.pt'):
    filepath = os.path.join(folder_path, filename)
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
    else:
        print(f"No checkpoint file found at {filepath}")


def compute_loss(output_logits, labels):
    return nn.CrossEntropyLoss()(output_logits.view(-1, output_logits.size(-1)), labels.view(-1))

def setup_training(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    return optimizer
