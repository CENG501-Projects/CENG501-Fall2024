import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import padding


# Adjusted crop_like function as per your suggestion
def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]

def convRelu(channels_in, channels_out, kernel_size=3, stride=2):
    layers =    nn.Sequential(
                    nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
                    nn.BatchNorm2d(channels_out),
                    nn.LeakyReLU(0.1, inplace=True)
                )
    return layers

def shapeConservedConvRelu(channels_in, channels_out, kernel_size=3, stride=1):
    layers =    nn.Sequential(
                    nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
                    nn.BatchNorm2d(channels_out),
                    nn.LeakyReLU(0.1, inplace=True)
                )
    return layers

def deConvRelu(channels_in, channels_out):
    layers =    nn.Sequential(
                    nn.ConvTranspose2d(channels_in, channels_out, kernel_size=4, stride=2, bias=False),
                    nn.BatchNorm2d(channels_out),
                    nn.LeakyReLU(0.1, inplace=True)
                )
    return layers


def predict_flow(channels_in):
    layer = nn.Sequential(
                nn.Conv2d(channels_in, 2, kernel_size=1, stride=1, padding=0, bias=False)
            )
    return layer

class EventFlow(nn.Module):
    def __init__(self):
        super(EventFlow, self).__init__()
        
        # ENCODER LAYERS
        self.encoder1 = convRelu(20, 64, kernel_size=3, stride=2)
        self.encoder2 = convRelu(64, 128, kernel_size=3, stride=2)
        self.encoder3 = convRelu(128, 256, kernel_size=3, stride=2)
        self.encoder4 = convRelu(256, 512, kernel_size=3, stride=2)
        
        # RESIDUAL LAYERS
        self.res11 = shapeConservedConvRelu(512, 512, kernel_size=3, stride=1)
        self.res12 = shapeConservedConvRelu(512, 512, kernel_size=3, stride=1)
        self.res21 = shapeConservedConvRelu(512, 512, kernel_size=3, stride=1)
        self.res22 = shapeConservedConvRelu(512, 512, kernel_size=3, stride=1)
        
        # DECODER LAYERS
        self.decoder4 = deConvRelu(512+2, 256)
        self.decoder3 = deConvRelu(256+256+2, 128)
        self.decoder2 = deConvRelu(128+128+2, 64)
        self.decoder1 = deConvRelu(64+64+2, 32)
        
        # FLOW PREDICTORS
        self.flow4 =  predict_flow(512)
        self.flow3 =  predict_flow(256)
        self.flow2 =  predict_flow(128)
        self.flow1 =  predict_flow(64)
        self.flow  =  predict_flow(32)
        

    def forward(self, x):
        # Zero padding to enables adaptive input size. 
        padded_tensor = padding(x)
        
        ########### ENCODER ###########
        e1 = self.encoder1(padded_tensor)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        ########### RESIDUAL ###########
        r11 = self.res11(e4)
        r12 = self.res12(r11) + e4
        r21 = self.res21(r12)
        r22 = self.res22(r21) + r12
        
        ########### DECODER ###########
        f4  = self.flow4(r22)
        cc4 = torch.cat((r22,f4),1)
        
        d3  = self.decoder4(cc4)
        d3  = crop_like(d3, e3)
        f3  = self.flow3(d3)
        cc3 = torch.cat((e3, d3, f3),1)

        d2  = self.decoder3(cc3)
        d2  = crop_like(d2, e2)
        f2  = self.flow2(d2)
        cc2 = torch.cat((e2, d2, f2),1)
        

        d1  = self.decoder2(cc2)
        d1  = crop_like(d1, e1)
        f1  = self.flow1(d1)
        cc1 = torch.cat((e1, d1, f1),1)

        d0      = self.decoder1(cc1)
        d0      = crop_like(d0, padded_tensor)
        flow    = self.flow(d0)
        
        # Return all scale level flows
        return flow, f1, f2, f3, f4