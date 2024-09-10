import torch
from torch import nn, optim
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # Ensure x is 3D: [batch_size, sequence_length, hidden_size]
        if x.dim() == 2:
            # Add batch dimension if x is 2D
            x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)  # [sequence_length, batch_size, hidden_size]
        output = self.transformer(x, x)
        output = output.permute(1, 0, 2)  # [batch_size, sequence_length, hidden_size]
        output = self.weighted_sum(output)
        return output

class SpeechToTextSummarizer(nn.Module):
    def __init__(self, llama_model_name='huggyllama/llama-7b', 
                 hidden_size=768, num_attention_heads=12, num_layers=6):
        super(SpeechToTextSummarizer, self).__init__()
        self.q_former = QFormer(hidden_size=hidden_size, num_attention_heads=num_attention_heads, num_layers=num_layers)
        self.q_former.to(device)  
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.text_tokenizer = LlamaTokenizer.from_pretrained(llama_model_name)
        llama_model = LlamaForCausalLM.from_pretrained(llama_model_name, quantization_config=bnb_config)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.text_generator = get_peft_model(llama_model, lora_config)
        self.text_generator.to(device)  
    
    def forward(self, audio_input, text_input=None):
        audio_input = audio_input.to(device)
        
        # If audio_input is 2D, convert to 3D by adding a batch dimension
        if audio_input.dim() == 2:
            audio_input = audio_input.unsqueeze(0)

        refined_features = self.q_former(audio_input)
        if text_input is None:
            input_ids = self.text_tokenizer("<s>", return_tensors="pt").input_ids.to(device)
        else:
            input_ids = self.text_tokenizer(text_input, return_tensors="pt").input_ids.to(device)
        output = self.text_generator(input_ids=input_ids, encoder_hidden_states=refined_features)
        
        return output.logits

