import torch
import torch.nn as nn
from utility import *

def train_sentence_level_asr(model, data_loader, optimizer):
    model.train()
    for batch in data_loader:
        if batch is None:
            continue

        audio_inputs = batch['audio_inputs']
        text_embeddings = batch['text_embeddings']

        optimizer.zero_grad()
        output_logits = model(audio_inputs)
        loss = compute_loss(output_logits, text_embeddings)
        loss.backward()
        optimizer.step()

def train_document_level_asr(model, data_loader, optimizer):
    model.train()
    for batch in data_loader:
        if batch is None:
            continue

        audio_inputs = batch['audio_inputs']
        summary_embeddings = batch['summary_embeddings']

        optimizer.zero_grad()
        output_logits = model(audio_inputs)
        loss = compute_loss(output_logits, summary_embeddings)
        loss.backward()
        optimizer.step()

def train_end_to_end_summarization(model, data_loader, optimizer):
    model.train()
    for batch in data_loader:
        if batch is None:
            continue

        audio_inputs = batch['audio_inputs']
        summaries = batch['summaries'] 

        optimizer.zero_grad()
        output_logits = model(audio_inputs)
        loss = compute_loss(output_logits, summaries)
        loss.backward()
        optimizer.step()

def train_model(model, data_loaders, num_epochs, checkpoint_folder='checkpoint'):
    optimizer = setup_training(model)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_sentence_level_asr(model, data_loaders['sentence_asr'], optimizer)
        train_document_level_asr(model, data_loaders['document_asr'], optimizer)
        train_end_to_end_summarization(model, data_loaders['summarization'], optimizer)
        save_model(model, optimizer, folder_path=checkpoint_folder)
