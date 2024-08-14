import os
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor
from pydub import AudioSegment
import torch
from torch.utils.data import DataLoader, random_split

class SpeechDataset(Dataset):
    def __init__(self, audio_dir, text_dir, summary_dir, processor=None):
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.summary_dir = summary_dir
        self.processor = processor if processor else Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        self.audio_files = sorted(os.listdir(audio_dir))
        self.text_files = sorted(os.listdir(text_dir))
        self.summary_files = sorted(os.listdir(summary_dir))
        
        assert len(self.audio_files) == len(self.text_files) == len(self.summary_files), "Mismatched dataset lengths!"
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        text_path = os.path.join(self.text_dir, self.text_files[idx])
        summary_path = os.path.join(self.summary_dir, self.summary_files[idx])
        audio = AudioSegment.from_wav(audio_path)
        audio_input = self.processor(audio.get_array_of_samples(), sampling_rate=audio.frame_rate, return_tensors="pt").input_values.squeeze()
        with open(text_path, 'r') as f:
            text = f.read().strip()
        with open(summary_path, 'r') as f:
            summary = f.read().strip()
        
        return {
            'audio_input': audio_input,
            'text': text,
            'summary': summary
        }


def get_data_loaders(batch_size=8, shuffle=True, audio_dir='audio', text_dir='text', summary_dir='summary', validation_split=0.2):
    dataset = SpeechDataset(audio_dir=audio_dir, text_dir=text_dir, summary_dir=summary_dir)
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, val_loader
