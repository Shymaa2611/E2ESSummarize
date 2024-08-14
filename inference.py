import torch
from transformers import LlamaTokenizer
from model import SpeechToTextSummarizer

def segment_audio(audio, sample_rate=16000, segment_duration=15, overlap=0.5):
    segment_samples = int(segment_duration * sample_rate)
    overlap_samples = int(overlap * segment_samples)
    segments = []

    start = 0
    while start < audio.size(1):
        end = start + segment_samples
        segment = audio[:, start:end]
        if segment.size(1) < segment_samples:
            segment = torch.nn.functional.pad(segment, (0, segment_samples - segment.size(1)))
        segments.append(segment)
        start = end - overlap_samples  
    return segments

def infer(model, audio_input, tokenizer, segment_duration=15, overlap=0.5):
    model.eval()
    audio_segments = segment_audio(audio_input, sample_rate=16000, segment_duration=segment_duration, overlap=overlap)
    features = []
    for segment in audio_segments:
        feature = model.speech_encoder(segment.unsqueeze(0))
        features.append(feature)
    concatenated_features = torch.cat(features, dim=1)
    
    with torch.no_grad():
        logits = model(concatenated_features)
        predicted_ids = torch.argmax(logits, dim=-1)
        generated_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        
        return generated_text


if __name__ == "__main__":
    audio_input = torch.randn(1, 16000 * 300)  
    model = SpeechToTextSummarizer()
    tokenizer = LlamaTokenizer.from_pretrained('huggingface/llama')
    summary_text = infer(model, audio_input, tokenizer)
    print(f"Generated Text: { summary_text}")
