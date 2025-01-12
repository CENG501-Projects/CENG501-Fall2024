import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quant
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import random
import numpy as np
import os


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

        self.add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.add.add(out, self.shortcut(x))
        out = self.relu2(out)
        return out


class QuantizableResNet20(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(QuantizableResNet20, self).__init__()
        self.in_planes = 16

        # First layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)

        # Main layers
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # Global pool + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*block.expansion, num_classes)

        # Quant stubs
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)  # Quantize input

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.dequant(x)  # Dequantize output
        return x

    def fuse_model(self):
        # Fuse the head
        torch.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)

        # Fuse each BasicBlock
        for m in self.modules():
            if isinstance(m, BasicBlock):
                torch.quantization.fuse_modules(m, ['conv1', 'bn1', 'relu1'], inplace=True)
                torch.quantization.fuse_modules(m, ['conv2', 'bn2'], inplace=True)
                if len(m.shortcut) == 2:
                    # Fuse conv+bn in shortcut
                    torch.quantization.fuse_modules(m.shortcut, ['0', '1'], inplace=True)


def create_quantizable_resnet20(num_classes=10):
    return QuantizableResNet20(BasicBlock, [3, 3, 3], num_classes=num_classes)
