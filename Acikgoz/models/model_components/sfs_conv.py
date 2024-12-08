import torch
import torch.nn as nn
import torch.nn.functional as F


# SPU: Spatial Perception Unit
class SPU(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super(SPU, self).__init__()
        self.groups = groups
        self.group_in_channels = in_channels // groups
        self.group_out_channels = out_channels // groups

        self.convs = nn.ModuleList([
            nn.Conv2d(self.group_in_channels, self.group_out_channels, kernel_size=3 + 2 * i, padding=1 + i, bias=False)
            for i in range(groups)
        ])
        self.align_conv = nn.ModuleList([
            nn.Conv2d(self.group_out_channels, self.group_in_channels, kernel_size=1, bias=False)
            for _ in range(groups)
        ])
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        splits = torch.split(x, self.group_in_channels, dim=1)
        outputs = []
        for i, conv in enumerate(self.convs):
            if i == 0:
                outputs.append(conv(splits[i]))
            else:
                aligned = self.align_conv[i](outputs[-1])
                if aligned.shape[2:] != splits[i].shape[2:]:
                    aligned = F.interpolate(aligned, size=splits[i].shape[2:], mode="bilinear", align_corners=False)
                outputs.append(conv(splits[i] + aligned))
        fused = torch.cat(outputs, dim=1)
        return self.final_conv(fused)


# FPU: Frequency Perception Unit
class FPU(nn.Module):
    def __init__(self, in_channels, out_channels, freq_scales=4):
        super(FPU, self).__init__()
        self.gabor_filters = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
            for _ in range(freq_scales)
        ])
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        features = sum(gabor(x) for gabor in self.gabor_filters)
        return self.final_conv(features)


# CSU: Channel Selection Unit
class CSU(nn.Module):
    def __init__(self, out_channels):
        super(CSU, self).__init__()
        self.fc = nn.Linear(2 * out_channels, 2)

    def forward(self, spatial, frequency):
        combined = torch.cat([spatial, frequency], dim=1)
        weights = F.adaptive_avg_pool2d(combined, 1).view(combined.size(0), -1)
        weights = F.softmax(self.fc(weights), dim=1)
        beta, gamma = weights[:, 0].view(-1, 1, 1, 1), weights[:, 1].view(-1, 1, 1, 1)
        return beta * spatial + gamma * frequency


# SFS-Conv: Space-Frequency Selection Convolution
class SFSConv(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.5, freq_scales=4):
        super(SFSConv, self).__init__()
        self.freq_channels = int(alpha * in_channels)
        self.spatial_channels = in_channels - self.freq_channels

        self.spu = SPU(self.spatial_channels, out_channels)
        self.fpu = FPU(self.freq_channels, out_channels, freq_scales)
        self.csu = CSU(out_channels)

    def forward(self, x):
        freq_input, spatial_input = torch.split(x, [self.freq_channels, self.spatial_channels], dim=1)
        spatial_features = self.spu(spatial_input)
        frequency_features = self.fpu(freq_input)
        return self.csu(spatial_features, frequency_features)


# Test the Fixed Code
if __name__ == "__main__":
    x = torch.randn(8, 64, 32, 32)  # Test input
    sfs_conv = SFSConv(in_channels=64, out_channels=128, alpha=0.5)
    y = sfs_conv(x)

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {y.shape}")