import os
import librosa
from torch.utils.data import Dataset

class AudioSummarizationDataset(Dataset):
    def __init__(self, audio_dir, summary_dir):
        self.audio_dir = audio_dir
        self.summary_dir = summary_dir
        self.audio_files = sorted(os.listdir(audio_dir))
        self.summary_files = sorted(os.listdir(summary_dir))
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        summary_path = os.path.join(self.summary_dir, self.summary_files[idx])
        audio, _ = librosa.load(audio_path, sr=16000)
        with open(summary_path, 'r') as file:
            summary = file.read().strip()
        
        return {'audio': audio, 'summary': summary}


dataset = AudioSummarizationDataset('data/audio',  'data/summary')

