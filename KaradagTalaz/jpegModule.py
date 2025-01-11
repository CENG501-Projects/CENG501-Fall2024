import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL.Image as Image
import jpegio as jio
from scipy.fftpack import dct, idct
from typing import List, Optional, Tuple
import torch_dct as dct
from copy import deepcopy
from cimd_utils import *

class JpegArtifactLearningModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=21, out_channels=21, kernel_size=3, dilation=8, padding=8)
        self.bn1 = nn.BatchNorm2d(21)
        self.conv2 = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.conv3 = nn.Conv2d(in_channels=64*3, out_channels=3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(3)
        
    
    def forward(self, q_tensor, R, binary_volume):
        
        bv = self.conv1(binary_volume)
        bv = self.bn1(bv)
        bv = F.relu(bv)
        bv = self.conv2(bv)
        bv = self.bn2(bv)
        bv = F.relu(bv)
        
        bv = bv.squeeze(1) 
        
        a1 = q_tensor * bv  
        a2 = bv 
        a3 = R * bv
        
        a1 = reshape_dct_blocks(a1)
        a2 = reshape_dct_blocks(a2)
        a3 = reshape_dct_blocks(a3)
        
        concat = torch.cat([a1, a2, a3], dim=-1).permute(0, 3, 1, 2)
        concat = self.conv3(concat)
        concat = self.bn3(concat)
        concat = F.relu(concat)

        b, h, w = q_tensor.shape
        if h == 2048:
            concat = F.interpolate(concat, size=(1024, 672), mode='bilinear') 
        elif w == 2048:
            concat = F.interpolate(concat, size=(672, 1024), mode='bilinear')
        return concat
    