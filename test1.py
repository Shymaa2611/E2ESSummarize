import torch
import torch.nn as nn


class WaveNet(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.conv1=nn.Conv1d(in_channels=input_size,kernel_size=3,out_channels=output_size)
        self.dilated_convs=nn.ModuleList()
        self.residual_blocks=nn.ModuleList()
        