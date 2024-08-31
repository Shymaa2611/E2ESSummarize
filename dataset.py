import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import Wav2Vec2Processor, BertTokenizer, BertModel
from pydub import AudioSegment
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class SpeechDataset(Dataset):
    def __init__(self, audio_dir, text_dir, summary_dir, processor=None, tokenizer=None, embedding_model=None, segment_length=15):
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.summary_dir = summary_dir
        self.processor = processor if processor else Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained("bert-base-uncased")
        self.embedding_model = embedding_model if embedding_model else BertModel.from_pretrained("bert-base-uncased")
        self.audio_files = sorted(os.listdir(audio_dir))
        self.text_files = sorted(os.listdir(text_dir))
        self.summary_files = sorted(os.listdir(summary_dir))
        self.segment_length = segment_length * 1000

        assert len(self.audio_files) == len(self.text_files) == len(self.summary_files), "Mismatched dataset lengths!"

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        text_path = os.path.join(self.text_dir, self.text_files[idx])
        summary_path = os.path.join(self.summary_dir, self.summary_files[idx])
        
        try:
            audio = AudioSegment.from_mp3(audio_path)
            audio = audio.set_frame_rate(16000)
        except Exception as e:
            print(f"Error loading or resampling audio file {audio_path}: {e}")
            return None
        
        audio_duration = len(audio)
        segments = []
        for start in range(0, audio_duration, self.segment_length):
            end = min(start + self.segment_length, audio_duration)
            segment = audio[start:end]
            segments.append(segment)
        
        audio_inputs = []
        for segment in segments:
            samples = np.array(segment.get_array_of_samples())
            audio_input = self.processor(samples, 
                                         sampling_rate=16000,
                                         return_tensors="pt").input_values.squeeze(0)
            audio_inputs.append(audio_input)
        
        audio_inputs = torch.cat(audio_inputs, dim=0)
        
        try:
            with open(text_path, 'r') as f:
                text = f.read().strip()
            with open(summary_path, 'r') as f:
                summary = f.read().strip()
        except Exception as e:
            print(f"Error loading text or summary file: {e}")
            return None
        text_tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        summary_tokens = self.tokenizer(summary, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            text_embeddings = self.embedding_model(**text_tokens).last_hidden_state.mean(dim=1)  # Mean pooling
            summary_embeddings = self.embedding_model(**summary_tokens).last_hidden_state.mean(dim=1)
        
        return {
            'audio_inputs': audio_inputs, 
            'text': text,
            'summary': summary,
            'text_embeddings': text_embeddings.squeeze(0), 
            'summary_embeddings': summary_embeddings.squeeze(0)
        }

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    audio_inputs = [item['audio_inputs'] for item in batch]
    texts = [item['text'] for item in batch]
    summaries = [item['summary'] for item in batch]
    text_embeddings = [item['text_embeddings'] for item in batch]
    summary_embeddings = [item['summary_embeddings'] for item in batch]
    audio_inputs = pad_sequence(audio_inputs, batch_first=True)
    text_embeddings = torch.stack(text_embeddings)
    summary_embeddings = torch.stack(summary_embeddings)
    
    return {
        'audio_inputs': audio_inputs,
        'texts': texts,
        'summaries': summaries,
        'text_embeddings': text_embeddings,
        'summary_embeddings': summary_embeddings
    }

def get_data_loaders(batch_size=8, shuffle=True, audio_dir='data/audio', text_dir='data/text', summary_dir='data/summary', validation_split=0.2):
    dataset = SpeechDataset(audio_dir=audio_dir, text_dir=text_dir, summary_dir=summary_dir)
    dataset_size = len(dataset)
    val_size = max(int(validation_split * dataset_size), 1)
    train_size = dataset_size - val_size
    
    if train_size == 0:
        raise ValueError("Training set size is 0. Consider reducing the validation split or increasing dataset size.")
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    return train_loader, val_loader

train_loader, val_loader = get_data_loaders()

for data in train_loader:
    if data:
        print(data) 
