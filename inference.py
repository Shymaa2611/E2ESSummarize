import torch
import librosa
from args import *



def summarize_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        features = processor(audio, return_tensors='pt').input_values.squeeze(0)
        features = torch.tensor(features).unsqueeze(0)  
        audio_features = s_encoder.extract_features(features)
    compressed_features = q_former(audio_features)
    input_ids = compressed_features
    summary_ids = llama2_model.generate(input_ids=input_ids, max_length=150, num_beams=4, early_stopping=True)
    summary = llama2_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary


audio_file_path = ''
summary = summarize_audio(audio_file_path)
print("Summary:", summary)
