import torch
from torch import nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model, LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, get_peft_model

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
    def __init__(self, hidden_size, num_attention_heads, num_layers):
        super(QFormer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=hidden_size, 
            nhead=num_attention_heads, 
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.weighted_sum = nn.Linear(hidden_size, hidden_size)  

    def forward(self, x):
        x = x.permute(1, 0, 2)  
        output = self.transformer(x, x)
        output = output.permute(1, 0, 2)
        output = self.weighted_sum(output)  
        return output


class SpeechToTextSummarizer(nn.Module):
    def __init__(self, llama_model_name='huggyllama/llama-7b', 
                 hidden_size=768, num_attention_heads=12, num_layers=6):
        super(SpeechToTextSummarizer, self).__init__()
        self.speech_encoder = SEncoder()
        self.q_former = QFormer(hidden_size=hidden_size, num_attention_heads=num_attention_heads, num_layers=num_layers)
        self.text_tokenizer = LlamaTokenizer.from_pretrained(llama_model_name, legacy=False)
        llama_model = LlamaForCausalLM.from_pretrained(llama_model_name)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.text_generator = get_peft_model(llama_model, lora_config)
    
    def forward(self, audio_input, segment_pos_embeds=None):
        speech_features = self.speech_encoder(audio_input)
        if segment_pos_embeds is not None:
            speech_features += segment_pos_embeds  
        refined_features = self.q_former(speech_features)  
        input_ids = self.text_tokenizer("<s>", return_tensors="pt").input_ids
        gpt_output = self.text_generator(input_ids=input_ids, encoder_hidden_states=refined_features)
        return gpt_output.logits

