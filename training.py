import torch
from torch.utils.data import DataLoader
from dataset import SpeechSummarizationDataset
from model import SpeechToTextSummarizer

def train(model, dataset, optimizer, criterion, num_epochs=5):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    model

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for segments, input_ids, attention_mask in dataloader:
            segments = segments
            input_ids = input_ids
            attention_mask = attention_mask

            optimizer.zero_grad()

            # Forward pass through each segment
            for segment in segments:
                segment = segment.unsqueeze(0) # Add batch dimension
                outputs = model(segment)
                
                # Compute loss (dummy loss for illustration)
                # In practice, you should compute loss based on your specific requirements
                loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

if __name__ == "__main__":
    # Define paths to your data directories
    speech_dir = 'data//audio'
    text_dir = 'data//summary'
    
    # Initialize dataset and model
    dataset = SpeechSummarizationDataset(speech_dir, text_dir)
    model = SpeechToTextSummarizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()  # Adjust as needed

    # Train the model
    train(model, dataset, optimizer, criterion)

    