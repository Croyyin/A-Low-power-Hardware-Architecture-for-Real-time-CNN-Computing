import torch.nn as nn
import torch.nn.functional as F
from math import floor
import torchvision.models as models

from utils.quantify import *


class VGGMini_CNN(nn.Module):
    def __init__(self):
        super(VGGMini_CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=0),                              
            nn.ReLU(),                     
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=0),                              
            nn.ReLU(),                     
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),               
        )

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


def test_modify():
    net = VGGMini_CNN()
    for name,param in net.named_parameters():
	    print(name,param)

if __name__ == "__main__":
    test_modify()