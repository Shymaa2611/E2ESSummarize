from model import SpeechToTextSummarizer
import torch
num_epochs=20
model = SpeechToTextSummarizer()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()  