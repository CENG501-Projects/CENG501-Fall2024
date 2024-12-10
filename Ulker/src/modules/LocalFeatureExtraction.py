import torch
import torch.nn as nn
from torchsummary import summary

class UNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(UNetFeatureExtractor, self).__init__()

        # Operations
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Encoder from Superpoint (VGG-Style).
        # Details are not published in paper, taken from Superpoint and VGG. 
        # Batch normalization is not used in Superpoint implementation, but used in VGG.
        channel_1, channel_2, channel_3, channel_4 = 64, 64, 128, 128
        self.encoder_block_1a = self.encoderBlock(1, channel_1)
        self.encoder_block_1b = self.encoderBlock(channel_1, channel_1)
        self.encoder_block_2a = self.encoderBlock(channel_1, channel_2)
        self.encoder_block_2b = self.encoderBlock(channel_2, channel_2)
        self.encoder_block_3a = self.encoderBlock(channel_2, channel_3)
        self.encoder_block_3b = self.encoderBlock(channel_3, channel_3)
        self.encoder_block_4a = self.encoderBlock(channel_3, channel_4)
        self.encoder_block_4b = self.encoderBlock(channel_4, channel_4)

        # Decoder
        self.decoder_block_1a = self.decoderBlockA(channel_4, channel_3)
        self.decoder_block_1b = self.decoderBlockB(channel_4, channel_3)
        self.decoder_block_2a = self.decoderBlockA(channel_3, channel_2)
        self.decoder_block_2b = self.decoderBlockB(channel_3, channel_2)
        self.decoder_block_3a = self.decoderBlockA(channel_2, channel_1)
        self.decoder_block_3b = self.decoderBlockB(channel_2, channel_1)

    def forward(self, x):
        
        # Input size: 1x864x864
        # Output sizes: a: 128x108x108 b: 128x216x216 c: 64x432x432 d: 64x864x864

        # Encoder
        # Sequentially: encoder_block: conv - batchNorm2D - ReLU
        x1 = self.encoder_block_1a(x)
        x1 = self.encoder_block_1b(x1)
        f1 = x1
        x1 = self.pool(x1) 
        # 1/2 Resolution
        x2 = self.encoder_block_2a(x1)
        x2 = self.encoder_block_2b(x2)
        f2 = x2
        x2 = self.pool(x2) 
        # 1/4 Resolution
        x3 = self.encoder_block_3a(x2)
        x3 = self.encoder_block_3b(x3)
        f3 = x3
        x3 = self.pool(x3) 
        # 1/8 Resolution
        x4 = self.encoder_block_4a(x3)
        x4 = self.encoder_block_4b(x4) 
        # 1/8 Resolution since no pool

        # Decoder
        # 1/8 Resolution
        y1 = self.decoder_block_1a(x4)
        y1 = torch.cat((y1, f3), dim=0)
        y1 = self.decoder_block_1b(y1)
        # 1/4 Resolution
        y2 = self.decoder_block_2a(y1)
        y2 = torch.cat((y2, f2), dim=0)
        y2 = self.decoder_block_2b(y2)
        # 1/2 Resolution
        y3 = self.decoder_block_3a(y2)
        y3 = torch.cat((y3, f1), dim=0)
        y3 = self.decoder_block_3b(y3)
        # 1 Resolution

        return [x4, y1, y2, y3]


    def encoderBlock(self, in_channels, out_channels):
        return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))


    def decoderBlockA(self, in_channels, out_channels):
        return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def decoderBlockB(self, in_channels, out_channels):
        return nn.Sequential(
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

model = UNetFeatureExtractor()
summary(model, (1, 864, 864))