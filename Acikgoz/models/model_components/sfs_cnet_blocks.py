import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. CBR Block (Convolution, BatchNorm, ReLU)
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# 3. Upsample and Concat operation (for multi-scale feature fusion)
class UpsampleConcat(nn.Module):
    def __init__(self, in_channels, skip_channels):
        super(UpsampleConcat, self).__init__()

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return torch.cat([x, skip], dim=1)

class DecoupledHead(nn.Module):
    def __init__(self, input_channels, num_boxes=64, num_classes=3, image_size=(640, 640)):
        """
        Decoupled Detection Head with Bounding Box Scaling and Constrained Predictions.
        Args:
            input_channels: Number of input feature map channels.
            num_boxes: Number of predicted boxes per location.
            num_classes: Number of object classes (excluding background).
            image_size: Tuple representing the dimensions of the input image (width, height).
        """
        super(DecoupledHead, self).__init__()
        self.input_channels = input_channels
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.image_size = image_size  # Image size to scale predictions (e.g., 640x640)

        # Shared feature refinement layers
        self.shared_conv = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        self.shared_bn = nn.BatchNorm2d(256)
        self.shared_activation = nn.ReLU()  # Use non-inplace ReLU

        # Bounding box regression branch
        self.bbox_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bbox_bn1 = nn.BatchNorm2d(128)
        self.bbox_conv2 = nn.Conv2d(128, 4 * num_boxes, kernel_size=1)

        # Classification branch
        self.cls_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.cls_bn1 = nn.BatchNorm2d(128)
        self.cls_conv2 = nn.Conv2d(128, num_classes * num_boxes, kernel_size=1)

    def forward(self, x):
        x = self.shared_conv(x)
        x = self.shared_bn(x)
        x = self.shared_activation(x)  # Non-inplace ReLU

        bbox = self.bbox_conv1(x)
        bbox = self.bbox_bn1(bbox)
        bbox = F.relu(bbox)  # Non-inplace ReLU
        bbox_pred = torch.sigmoid(self.bbox_conv2(bbox)).view(x.size(0), -1, 4)

        cls = self.cls_conv1(x)
        cls = self.cls_bn1(cls)
        cls = F.relu(cls)  # Non-inplace ReLU
        cls_pred = self.cls_conv2(cls).view(x.size(0), -1, self.num_classes)

        return bbox_pred, cls_pred




