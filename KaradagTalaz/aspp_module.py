import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """
    Args:
        in_channels (int): 
        out_channels (int): 
        rates (tuple): dilation rates
    """
    def __init__(self, in_channels, out_channels, rates=(3, 6, 9)):
        super(ASPP, self).__init__()
        
        # 1x1 convolution branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # dilated convolution branches
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                         padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for rate in rates
        ])
        
        # global Average Pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # final 1x1 convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 
                     kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        _, _, h, w = x.size()
        
        conv1x1_output = self.conv1x1(x)
        
        dilated_outputs = [conv(x) for conv in self.dilated_convs]
        
        gap_output = self.global_avg_pool(x)
        gap_output = F.interpolate(gap_output, size=(h, w), 
                                 mode='bilinear', align_corners=False)
        
        concat_features = torch.cat(
            [conv1x1_output] + dilated_outputs + [gap_output], 
            dim=1
        )
        
        output = self.final_conv(concat_features)
        
        return output
