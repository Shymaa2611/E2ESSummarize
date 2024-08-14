import torch
import os

def train(model, train_loader, optimizer, criterion, num_epochs=5, checkpoint_dir='checkpoint', device='cpu'):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for segments, input_ids, attention_mask in train_loader:
            segments = [seg.to(device) for seg in segments]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            features = [model.speech_encoder(segment.unsqueeze(0)) for segment in segments]
            concatenated_features = torch.cat(features, dim=1)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(concatenated_features)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss}")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, checkpoint_path)
    print(f'checkpoint saved to {checkpoint_path}')

    print('Training complete.')
