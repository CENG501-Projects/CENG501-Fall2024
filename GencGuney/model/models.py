# src/models/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# 1. ResNet for CIFAR
#    Adjusted from the official PyTorch ImageNet ResNet code to work on 32x32 inputs.
# --------------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CIFARResNet(nn.Module):
    """
    A generic ResNet implementation for CIFAR-like 32x32 inputs.
    We remove the initial max-pool and adjust the first conv to stride=1.
    """
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_planes = 64

        # NOTE: For CIFAR, we use a smaller kernel, stride=1, no max-pool.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # if dimension changes (stride != 1 or in_planes != planes*block.expansion),
        # we need a projection
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(num_classes=10):
    return CIFARResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=10):
    return CIFARResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=10):
    return CIFARResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes=10):
    return CIFARResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

def resnet152(num_classes=10):
    return CIFARResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


# --------------------------------------------------------------------------------
# 2. DenseNet for CIFAR
#    Adjusted from the official PyTorch DenseNet to accommodate 32x32 inputs.
# --------------------------------------------------------------------------------

class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1,
                               stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size, drop_rate):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layer = _DenseLayer(in_channels, growth_rate, bn_size, drop_rate)
            layers.append(layer)
            in_channels += growth_rate
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x


class CIFARDenseNet(nn.Module):
    """
    DenseNet implementation for CIFAR (e.g., DenseNet-121).
    We reduce the initial stride/pool to handle 32x32 without losing too much spatial info.
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=10):
        super().__init__()

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            # Typically, ImageNet DenseNet might have a pooling step here,
            # but for CIFAR, we omit it (or you can keep a small one).
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            denseblock = _DenseBlock(num_layers, num_features,
                                     growth_rate=growth_rate,
                                     bn_size=bn_size,
                                     drop_rate=drop_rate)
            self.features.add_module(f'denseblock{i+1}', denseblock)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:  # no transition layer after the last block
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet121(num_classes=10, growth_rate=32, drop_rate=0):
    """
    CIFAR variant of DenseNet-121.
    By default, uses block_config=(6,12,24,16) which is standard 121-layers,
    growth_rate=32, no dropout.
    """
    return CIFARDenseNet(
        growth_rate=growth_rate,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=drop_rate,
        num_classes=num_classes
    )


# --------------------------------------------------------------------------------
# 3. Model Creation Function
#    Helps you create one of these models by name for training or inference.
# --------------------------------------------------------------------------------

def build_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    Create a CIFAR-friendly model by name.
    
    model_name can be one of:
      'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121'
    """
    model_name = model_name.lower()
    if model_name == 'resnet18':
        return resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        return resnet34(num_classes=num_classes)
    elif model_name == 'resnet50':
        return resnet50(num_classes=num_classes)
    elif model_name == 'resnet101':
        return resnet101(num_classes=num_classes)
    elif model_name == 'resnet152':
        return resnet152(num_classes=num_classes)
    elif model_name == 'densenet121':
        return densenet121(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name {model_name}")
    

# --------------------------------------------------------------------------------
# Usage Example (if run directly):
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage: create ResNet18 for CIFAR-10
    model = build_model("resnet18", num_classes=10)
    print(model)

    # Example usage: create DenseNet-121
    densenet = build_model("densenet121", num_classes=10)
    print(densenet)

    # Test forward pass with a random input (batch_size=4, 3x32x32 images)
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    print("ResNet18 output shape:", output.shape)

    output_dense = densenet(x)
    print("DenseNet-121 output shape:", output_dense.shape)
