import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor, AutoModelForCausalLM, AutoTokenizer

class SEncoder(nn.Module):
    def __init__(self, model_name='facebook/wav2vec2-base-960h'):
        super(SEncoder, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
    
    def forward(self, audio_input):
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state

class QFormer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QFormer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class LLaMA2WithLoRA(nn.Module):
    def __init__(self, model_name='huggyllama/llama-7b'):
        super(LLaMA2WithLoRA, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

class E2ESpeechSummarization(nn.Module):
    def __init__(self, s_encoder, q_former, llama2):
        super(E2ESpeechSummarization, self).__init__()
        self.s_encoder = s_encoder
        self.q_former = q_former
        self.llama2 = llama2
    
    def forward(self, audio_input):
        audio_features = self.s_encoder(audio_input)
        compressed_features = self.q_former(audio_features)
        inputs = compressed_features.squeeze(0) 
        input_ids = inputs.long() 
        summary_ids = self.llama2.model.generate(input_ids=input_ids, max_length=150, num_beams=4, early_stopping=True)
        summary = self.llama2.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary
