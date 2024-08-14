import os
import torchaudio
from torch.utils.data import Dataset

class SummarizeDataset(Dataset):
    def __init__(self, text_folder, summary_folder, audio_folder):
        self.text_folder = text_folder
        self.summary_folder = summary_folder
        self.audio_folder = audio_folder
        self.texts = self.load_files(text_folder)
        self.summaries = self.load_files(summary_folder)
        self.audio_files = self.load_audio_files(audio_folder)
        self.common_files = set(self.texts.keys()) & set(self.summaries.keys()) & set(self.audio_files.keys())
    
    def load_files(self, folder):
        files = {}
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename), 'r') as file:
                    files[filename] = file.read().strip()
        return files
    
    def load_audio_files(self, folder):
        files = {}
        for filename in os.listdir(folder):
            if filename.endswith(".wav"):
                waveform, sample_rate = torchaudio.load(os.path.join(folder, filename))
                files[filename] = (waveform, sample_rate)
        return files
    
    def __len__(self):
        return len(self.common_files)
    
    def __getitem__(self, idx):
        filename = list(self.common_files)[idx]
        text = self.texts[filename]
        summary = self.summaries[filename]
        waveform, sample_rate = self.audio_files[filename]
        return text, summary, waveform, sample_rate
