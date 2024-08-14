import torch
from transformers import LlamaTokenizer

def infer(model, audio_input, tokenizer):
    model.eval()
    with torch.no_grad():
        logits = model(audio_input)
        # Generate text from logits
        predicted_ids = torch.argmax(logits, dim=-1)
        generated_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        return generated_text

# Example usage


# Inference example
audio_input = torch.randn(1, 16000 * 30)  # Dummy audio input
generated_text = infer(model, audio_input, LlamaTokenizer.from_pretrained('huggingface/llama'))
print(f"Generated Text: {generated_text}")