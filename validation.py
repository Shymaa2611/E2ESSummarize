import torch
from args import *

def validate(model,val_loader):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for segments, input_ids, attention_mask in val_loader:
                segments = segments.squeeze(0)
                audio_segments = [seg for seg in segments]

                features = []
                for segment in audio_segments:
                    feature = model.speech_encoder(segment.unsqueeze(0))
                    features.append(feature)

                concatenated_features = torch.cat(features, dim=1)

                outputs = model(concatenated_features)

                loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}")
