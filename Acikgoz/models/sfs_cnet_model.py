import torch
import torch.nn as nn

from models.model_components.sfs_cnet_blocks import CBR, UpsampleConcat, DecoupledHead
from models.model_components.sfs_conv import SFSConv

class SFSCNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, base_channels=32):
        super(SFSCNet, self).__init__()
        # Stage 1: Downsampling and initial feature extraction
        self.stage1 = nn.Sequential(
            CBR(in_channels, base_channels, kernel_size=6, stride=2, padding=2),
            CBR(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            SFSConv(base_channels, base_channels * 2)  # SFS-Conv with channel doubling
        )

        # Stage 2: Feature extraction with downsampling
        self.stage2 = nn.Sequential(
            CBR(base_channels * 2, base_channels * 4, kernel_size=3, stride=2),
            SFSConv(base_channels * 4, base_channels * 4)
        )

        # Stage 3: Further downsampling and feature extraction
        self.stage3 = nn.Sequential(
            CBR(base_channels * 4, base_channels * 8, kernel_size=3, stride=2),
            SFSConv(base_channels * 8, base_channels * 8)
        )

        # Stage 4: Final downsampling and feature extraction
        self.stage4 = nn.Sequential(
            CBR(base_channels * 8, base_channels * 16, kernel_size=3, stride=2),
            SFSConv(base_channels * 16, base_channels * 16)
        )

        # Upsampling and Concatenation blocks for multi-scale feature fusion
        self.up_concat1 = UpsampleConcat(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up_concat2 = UpsampleConcat(base_channels * 8, base_channels * 4, base_channels * 4)

        # Decoupled head for detection tasks
        self.head = DecoupledHead(base_channels * 4, num_classes)

    def forward(self, x):
        # Forward pass through stages
        skip1 = self.stage1(x)  # First downsampled feature map (skip connection)
        skip2 = self.stage2(skip1)  # Second downsampled feature map (skip connection)
        skip3 = self.stage3(skip2)  # Third downsampled feature map (skip connection)
        x = self.stage4(skip3)  # Fourth downsampled feature map

        # Upsampling and concatenating with skip connections
        x = self.up_concat1(x, skip3)
        x = self.up_concat2(x, skip2)

        # Decoupled head outputs bounding boxes and class predictions
        bboxes, classes = self.head(x)
        return bboxes, classes

if __name__ == "__main__":
    x = torch.randn(8, 3, 256, 256)
    model = SFSCNet(in_channels=3, num_classes=10, base_channels=32)
    bboxes, classes = model(x)
    print(f"Input Shape: {x.shape}")
    print(f"Bounding Box Output Shape: {bboxes.shape}")
    print(f"Class Output Shape: {classes.shape}")