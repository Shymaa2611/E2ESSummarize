from dataset import SpeechDataset,get_data_loaders
from torch.utils.data import DataLoader
from model import SpeechToTextSummarizer
from utility import load_model,setup_training
from training import train_model



def main(audio_dir,text_dir,summary_dir):
      # Example usage
   model = SpeechToTextSummarizer()
   optimizer = setup_training(model)
   checkpoint_folder = 'checkpoint'
   load_model(model, optimizer, folder_path=checkpoint_folder)
   num_epochs = 10  
   data_loaders,_=get_data_loaders(audio_dir,text_dir,summary_dir)
   train_model(model, data_loaders, num_epochs, checkpoint_folder=checkpoint_folder)
 

if __name__ == "__main__":
      audio_dir="data//audio"
      text_dir="data//text"
      summary_dir="data//summary"
      main(audio_dir,text_dir,summary_dir)
    
