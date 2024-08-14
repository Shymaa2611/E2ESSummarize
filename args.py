from dataset import *
from sklearn.model_selection import train_test_split
from model import *
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, AutoTokenizer, AutoModelForCausalLM

train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=32, shuffle=True)
num_epoch=20
s_encoder = SEncoder()
q_former = QFormer(input_dim=768, output_dim=512)  
llama2 = LLaMA2WithLoRA()
e2e_model = E2ESpeechSummarization(s_encoder, q_former, llama2)
optimizer = torch.optim.Adam(e2e_model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()  
WAV2VEC2_MODEL_NAME = 'facebook/wav2vec2-base-960h'
LLAMA2_MODEL_NAME = 'huggyllama/llama-7b'
processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_NAME)
llama2_model = AutoModelForCausalLM.from_pretrained(LLAMA2_MODEL_NAME)
llama2_tokenizer = AutoTokenizer.from_pretrained(LLAMA2_MODEL_NAME)
q_former = QFormer(input_dim=768, output_dim=512)

