import torch
from model import SpeechToTextSummarizer
import torch
from torch import nn, optim
from dataset import get_data_loaders
from utility import save_model

def stage_1_train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch in train_loader:
        audio_input, text_target = batch
        audio_input = audio_input.to(device)
        text_target = text_target.to(device)
        
        optimizer.zero_grad()
        logits = model(audio_input)
        loss = criterion(logits.view(-1, logits.size(-1)), text_target.view(-1))
        loss.backward()
        optimizer.step()

def stage_2_train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch in train_loader:
        audio_input, text_target = batch
        audio_input = audio_input.to(device)
        text_target = text_target.to(device)
        optimizer.zero_grad()
        audio_input = audio_input.flatten(1, 2)
        logits = model(audio_input)
        mask = (torch.rand_like(audio_input) > 0.15).float()
        audio_input *= mask
        
        loss = criterion(logits.view(-1, logits.size(-1)), text_target.view(-1))
        loss.backward()
        optimizer.step()

def stage_3_train(model, train_loader, optimizer, criterion, device, curriculum_steps=10):
    model.train()
    for step in range(curriculum_steps):
        for batch in train_loader:
            audio_input, text_target = batch
            audio_input = audio_input.to(device)
            text_target = text_target.to(device)
            
            optimizer.zero_grad()
            
            # Gradually remove text features
            if step < curriculum_steps - 1:
                logits = model(audio_input, text_input=text_target)
            else:
                logits = model(audio_input)

            loss = criterion(logits.view(-1, logits.size(-1)), text_target.view(-1))
            loss.backward()
            optimizer.step()


def train():
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     model = SpeechToTextSummarizer().to(device)
     optimizer = optim.Adam(model.parameters(), lr=1e-4)
     criterion = nn.CrossEntropyLoss()
     train_loader,_=get_data_loaders()
     stage_1_train(model, train_loader, optimizer, criterion, device)
     stage_2_train(model, train_loader, optimizer, criterion, device)
     stage_3_train(model, train_loader, optimizer, criterion, device)
     save_model(model,optimizer,folder_path='checkpoint')



