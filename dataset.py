import os
import torch
import torchaudio
from torch.utils.data import Dataset, random_split
from transformers import LlamaTokenizer

class SpeechSummarizationDataset(Dataset):
    def __init__(self, speech_dir, summary_dir, segment_duration=15, sample_rate=16000, overlap=0.5):
        self.speech_dir = speech_dir
        self.text_dir = summary_dir
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.overlap = overlap
        self.audio_files = [f for f in os.listdir(speech_dir) if f.endswith('.mp3')]
        self.tokenizer = LlamaTokenizer.from_pretrained("huggingface/llama") 

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.speech_dir, self.audio_files[idx])
        audio, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            audio = resampler(audio)
        segments = self.segment_audio(audio)
        text_path = os.path.join(self.text_dir, os.path.splitext(self.audio_files[idx])[0] + '.txt')
        with open(text_path, 'r') as file:
            summary_text = file.read().strip()
        summary_tokens = self.tokenizer(summary_text, return_tensors='pt', padding=True, truncation=True)

        return segments, summary_tokens['input_ids'].squeeze(0), summary_tokens['attention_mask'].squeeze(0)

    def segment_audio(self, audio):
        segment_samples = int(self.segment_duration * self.sample_rate)
        overlap_samples = int(self.overlap * segment_samples)
        segments = []

        start = 0
        while start < audio.size(1):
            end = start + segment_samples
            segment = audio[:, start:end]
            if segment.size(1) < segment_samples:
                segment = torch.nn.functional.pad(segment, (0, segment_samples - segment.size(1)))
            segments.append(segment)
            start = end - overlap_samples  
        return torch.stack(segments)

def split_dataset(dataset, train_size=0.8):
    num_samples = len(dataset)
    num_train = int(train_size * num_samples)
    num_val = num_samples - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    return train_dataset, val_dataset
