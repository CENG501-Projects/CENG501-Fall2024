import math
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
            nn.Conv2d(
                self.group_in_channels,
                self.group_out_channels,
                kernel_size=3 + 2 * i,
                padding=(3 + 2 * i) // 2,
                bias=False
            ) for i in range(groups)
        ])
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        splits = torch.split(x, self.group_in_channels, dim=1)
        outputs = []
        for i in range(self.groups):
            if i == 0:
                outputs.append(self.convs[i](splits[i]))
            else:
                outputs.append(self.convs[i](splits[i]) + outputs[-1])
        fused = torch.cat(outputs, dim=1)
        return self.final_conv(fused)

# Gabor Filter Function
def gabor_filter(kernel_size, sigma, theta, lambd, gamma, psi):
    k = kernel_size // 2
    y, x = torch.meshgrid(
        torch.arange(-k, k + 1, dtype=torch.float32),
        torch.arange(-k, k + 1, dtype=torch.float32),
        indexing='ij'
    )
    theta = torch.tensor(theta, dtype=torch.float32)
    x_rot = x * torch.cos(theta) + y * torch.sin(theta)
    y_rot = -x * torch.sin(theta) + y * torch.cos(theta)
    return torch.exp(-(x_rot**2 + gamma**2 * y_rot**2) / (2 * sigma**2)) * torch.cos(
        2 * torch.pi * x_rot / lambd + psi
    )

# FrFT Kernel Function
def frft_kernel(width, alpha, scale, sampling_interval):
    k = torch.arange(width, dtype=torch.float32)
    freq = k / (scale * sampling_interval)
    return torch.exp(-1j * math.pi * alpha * freq**2)

# Fractional Gabor Transform
def fractional_gabor_transform(input_tensor, frft_kernel, gabor_filter):
    batch_size, channels, height, width = input_tensor.shape

    if frft_kernel.size(0) != width:
        raise ValueError(f"FrFT kernel size {frft_kernel.size(0)} does not match input width {width}.")

    device = input_tensor.device  # Get the device of the input tensor
    frft_kernel = frft_kernel.to(device)  # Ensure the kernel is on the same device
    gabor_filter = gabor_filter.to(device)  # Ensure the Gabor filter is on the same device

    frft_kernel_resized = frft_kernel.view(1, 1, 1, width)

    frft_output = torch.fft.fft(input_tensor, dim=-1)
    frft_output = frft_output * frft_kernel_resized
    frft_output = torch.fft.ifft(frft_output, dim=-1).real

    gabor_filter_expanded = gabor_filter.unsqueeze(0).unsqueeze(0)  # [1, 1, kernel_size, kernel_size]

    gabor_output = F.conv2d(
        frft_output,
        gabor_filter_expanded.expand(channels, 1, *gabor_filter.size()),  # Expand for each channel
        groups=channels,  # Independent convolution per channel
        padding=gabor_filter.size(0) // 2
    )
    return gabor_output

# FPU: Frequency Perception Unit
class FPU(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales, num_orientations, alpha=0.8):
        super(FPU, self).__init__()
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.alpha = alpha
        self.out_channels = out_channels
        self.pwc = nn.Conv2d(in_channels * num_scales * num_orientations, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        outputs = []
        for scale in range(1, self.num_scales + 1):
            frft_kernel_dynamic = frft_kernel(width, self.alpha, scale, sampling_interval=1.0)
            for orientation in range(self.num_orientations):
                gabor_filter_dynamic = gabor_filter(
                    kernel_size=width,
                    sigma=4.0,
                    theta=orientation * torch.pi / self.num_orientations,
                    lambd=10.0,
                    gamma=0.5,
                    psi=0.0
                )
                outputs.append(fractional_gabor_transform(x, frft_kernel_dynamic, gabor_filter_dynamic))
        concatenated = torch.cat(outputs, dim=1)
        return self.pwc(concatenated)

class CSU(nn.Module):
    def __init__(self):
        super(CSU, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.softmax = nn.Softmax(dim=1)   # Softmax for channel-wise attention

    def forward(self, y_s, y_f):
        s_s = self.gap(y_s)  # Shape: [Batch_Size, Channels, 1, 1]
        s_f = self.gap(y_f)  # Shape: [Batch_Size, Channels, 1, 1]

        statistics = torch.cat([s_s, s_f], dim=1)  # Shape: [Batch_Size, 2*Channels, 1, 1]
        attention = self.softmax(statistics)  # Shape: [Batch_Size, 2*Channels, 1, 1]
        gamma, beta = torch.split(attention, [y_s.shape[1], y_f.shape[1]], dim=1)  # Shapes: [Batch_Size, Channels, 1, 1]
        y_fused = torch.cat([gamma * y_s, beta * y_f], dim=1)   # Shape: [Batch_Size, Channels, Height, Width]

        return y_fused

# SFS-Conv: Space-Frequency Selection Convolution
class SFSConv(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.5, spu_groups=4, fpu_scales=4, fpu_orientations=8, alpha_fpu=0.8):
        super(SFSConv, self).__init__()
        self.alpha = alpha
        self.spatial_channels = int((1 - alpha) * in_channels)
        self.frequency_channels = in_channels - self.spatial_channels

        # Point-wise convolutions to split channels
        self.spatial_pwc = nn.Conv2d(self.spatial_channels, self.spatial_channels, kernel_size=1, bias=False)
        self.frequency_pwc = nn.Conv2d(self.frequency_channels, self.frequency_channels, kernel_size=1, bias=False)

        # SPU and FPU
        self.spu = SPU(self.spatial_channels, self.spatial_channels, groups=spu_groups)
        self.fpu = FPU(self.frequency_channels, self.frequency_channels, fpu_scales, fpu_orientations, alpha_fpu)

        # CSU: Combines outputs from SPU and FPU
        self.csu = CSU()

        # Fusion convolution with dynamically calculated input channels
        self.fusion_conv = nn.Conv2d(
            in_channels,  # Total input channels
            out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        # Split input into spatial and frequency parts
        spatial_input, frequency_input = torch.split(x, [self.spatial_channels, self.frequency_channels], dim=1)

        # Process spatial and frequency features
        spatial_features = self.spatial_pwc(spatial_input)
        frequency_features = self.frequency_pwc(frequency_input)
        spatial_refined = self.spu(spatial_features)
        frequency_refined = self.fpu(frequency_features)

        # Fuse features using CSU
        fused_features = self.csu(spatial_refined, frequency_refined)
        # Pass fused features through fusion convolution
        return self.fusion_conv(fused_features)

# Test the implementation
if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 64, 64)
    sfs_conv = SFSConv(32, 32, alpha=0.5, spu_groups=4, fpu_scales=4, fpu_orientations=8, alpha_fpu=0.8)
    output = sfs_conv(input_tensor)
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output.shape)