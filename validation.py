import torch
from args import criterion

def validate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            audio = batch['audio']
            transcripts = batch['transcript']
            summaries = batch['summary']
            
            outputs = model(audio)
            loss = criterion(outputs, summaries)
            total_loss += loss.item()
    return total_loss / len(dataloader)