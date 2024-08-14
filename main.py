from args import *
from dataset import SpeechSummarizationDataset,split_dataset
from torch.utils.data import DataLoader
from training import train

def main(speech_dir,summary_dir):
    dataset = SpeechSummarizationDataset(speech_dir, summary_dir)
    train_dataset, val_dataset = split_dataset(dataset, train_size=0.8)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    train(model, train_loader, val_loader, optimizer, criterion)
    



if __name__ == "__main__":

    speech_dir = 'data/audio'
    summary_dir = 'data/summary'
    main()
   
  
  
    
