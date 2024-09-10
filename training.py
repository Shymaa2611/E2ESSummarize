import torch
from model import SpeechToTextSummarizer
from torch import nn, optim
from dataset import get_data_loaders
from utility import save_model

def stage_1_train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch in train_loader:
        audio_input = batch['audio_inputs'].to(device)
        text_embeddings = batch['text_embeddings'].to(device)
        summary_embeddings = batch['summary_embeddings'].to(device)
        
        optimizer.zero_grad()
        logits = model(audio_input, text_embeddings, summary_embeddings)
        loss = criterion(logits,text_embeddings)
        loss.backward()
        optimizer.step()

def stage_2_train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch in train_loader:
        audio_input = batch['audio_inputs'].to(device)
        text_embeddings = batch['text_embeddings'].to(device)
        summary_embeddings = batch['summary_embeddings'].to(device)
        
        optimizer.zero_grad()
        audio_input = audio_input.flatten(start_dim=1)
        text_embeddings = text_embeddings.flatten(start_dim=1)
        mask = (torch.rand_like(audio_input) > 0.15).float()
        audio_input *= mask
        text_mask = (torch.rand_like(text_embeddings) > 0.15).float()
        text_embeddings *= text_mask

        logits = model(audio_input, text_embeddings, summary_embeddings)
        
        loss = criterion(logits, summary_embeddings)
        loss.backward()
        optimizer.step()

def stage_3_train(model, train_loader, optimizer, criterion, device, curriculum_steps=10):
    model.train()
    for step in range(curriculum_steps):
        for batch in train_loader:
            audio_input = batch['audio_inputs'].to(device)
            text_embeddings = batch['text_embeddings'].to(device)
            summary_embeddings = batch['summary_embeddings'].to(device)
            
            optimizer.zero_grad()
            if step < curriculum_steps - 1:
                logits = model(audio_input, text_embeddings, summary_embeddings)
            else:
                logits = model(audio_input)

            loss = criterion(logits, summary_embeddings)
            loss.backward()
            optimizer.step()

def train():
    print("Start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeechToTextSummarizer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()  
    train_loader, _ = get_data_loaders()
    
    
    stage_1_train(model, train_loader, optimizer, criterion, device)
    stage_2_train(model, train_loader, optimizer, criterion, device)
    stage_3_train(model, train_loader, optimizer, criterion, device)
    
    save_model(model, optimizer, folder_path='checkpoint')

if __name__ == "__main__":
    train()
