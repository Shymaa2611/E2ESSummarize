import torch
import torch.nn  as nn

def train_sentence_level_asr(model, data_loader, optimizer):
    model.train()
    for batch in data_loader:
        audio_input = batch['audio_input']
        optimizer.zero_grad()
        output_logits = model(audio_input)
        loss = compute_loss(output_logits, batch['labels'])
        loss.backward()
        optimizer.step()

def train_document_level_asr(model, data_loader, optimizer):
    model.train()
    for batch in data_loader:
        audio_input = batch['audio_input']
        transcription_features = batch['transcription_features']
        optimizer.zero_grad()
        output_logits = model(audio_input)
        loss = compute_loss(output_logits, transcription_features)
        loss.backward()
        optimizer.step()


def train_end_to_end_summarization(model, data_loader, optimizer):
    model.train()
    for batch in data_loader:
        audio_input = batch['audio_input']
        optimizer.zero_grad()
        output_logits = model(audio_input)
        loss = compute_loss(output_logits, batch['summarization_labels'])
        loss.backward()
        optimizer.step()

def compute_loss(output_logits, labels):
    return nn.CrossEntropyLoss()(output_logits.view(-1, output_logits.size(-1)), labels.view(-1))
def setup_training(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    return optimizer

def train_model(model, data_loaders):
    optimizer = setup_training(model)    
    train_sentence_level_asr(model, data_loaders['sentence_asr'], optimizer)
    train_document_level_asr(model, data_loaders['document_asr'], optimizer)
    train_end_to_end_summarization(model, data_loaders['summarization'], optimizer)
