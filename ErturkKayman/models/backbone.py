import torch
import torch.nn as nn
from torchvision.models import resnet34

class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load a pre-trained ResNet-34 model
        self.resnet = resnet34(pretrained=pretrained)
        # Remove fully connected layers (only use convolutional features)
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        # Final output channel size (ResNet-34 outputs 512 channels)
        self.out_channels = 512

    def forward(self, x):
        return self.feature_extractor(x)

