import torch
from args import *
from validation import validate

def train(model):
    train_losses = []
    for epoch in range(num_epoch):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            audio = batch['audio']
            summaries = batch['summary']
            
            optimizer.zero_grad()
            outputs = model(audio, summary=summaries)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epoch}, Loss: {avg_loss}")
    return train_losses

def main():
  checkpoint_dir = 'checkpoint'
  if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
  for epoch in range(num_epoch):
     train(e2e_model, train_loader, optimizer, criterion)
     val_loss = validate(e2e_model,val_loader)
     print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
  checkpoint_path = os.path.join(checkpoint_dir,"model.pt")
  torch.save(e2e_model.state_dict(), checkpoint_path)

if __name__=="__main__":
    main()
