import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, List, Tuple
from jpeg_artifacts_module import CompressionArtifactsLearning
from aspp_module import ASPP

class ChannelAttention(nn.Module):
    def __init__(self, f_n_channel, f_n1_channel, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        
        self.conv1x1 = nn.Conv2d (in_channels=f_n1_channel, out_channels=f_n_channel, kernel_size=1)

        self.fc = nn.Sequential(
            nn.Linear(f_n_channel, f_n_channel // reduction_ratio),
            nn.ReLU(),
            nn.Linear(f_n_channel // reduction_ratio, f_n_channel),
            nn.Sigmoid()
        )

    def forward(self, f_n: torch.tensor, f_n1):
        f_n1 =  self.conv1x1(f_n1)
        f_n1 = f_n1.mean([2,3])
        f_n1 = self.fc(f_n1)
        f_n1 = f_n1.view(*f_n1.size(), 1 ,1)
        return f_n1 * f_n           

class SpatialAttention(nn.Module):
    def __init__(self, scale_factor):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=1)
        self.scale_factor = scale_factor
        
    def forward(self, fn, fn_1):
        # x: (B, C, H, W)
        # (B, 1, H, W)
        avg_pool = torch.mean(fn, dim=1, keepdim=True)
        # (B, 1, H, W)
        max_pool, _ = torch.max(fn, dim=1, keepdim=True)
        # (B, 2, H, W)
        y = torch.cat([avg_pool, max_pool], dim=1)
        # (B, 1, H, W)
        y = self.conv(y)
        # (B, C, H, W)
        fn_1 = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')(fn_1)
        return torch.sigmoid(y) * fn_1

class TwoBranchNetwork(nn.Module):
    def __init__(self, c1, c2,c3,c4):
        super(TwoBranchNetwork, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        
        self.rgb_hrnet = timm.create_model('hrnet_w18', pretrained=True)
        self.rgb_hrnet.eval()
        for param in self.rgb_hrnet.parameters():
            param.requires_grad = False 

        self.freq_stream = CompressionArtifactsLearning()
        
        self.rgb_aspp = nn.ModuleList([
            ASPP(c1, c1),
            ASPP(c2, c2),
            ASPP(c3, c3),
            ASPP(c4, c4)
        ])
        
        self.freq_aspp = nn.ModuleList([
            ASPP(c2, c2),
            ASPP(c3, c3),
            ASPP(c4, c4)
        ])
        
        self.rgb_channel_attention = nn.ModuleList([
            ChannelAttention(c1,c2),
            ChannelAttention(c2,c3),
            ChannelAttention(c3,c4)
        ])
        
        self.rgb_spatial_attention = nn.ModuleList([
            SpatialAttention(scale_factor=2),
            SpatialAttention(scale_factor=4),
            SpatialAttention(scale_factor=8)
        ])
        
        self.freq_channel_attention = nn.ModuleList([
            ChannelAttention(c2, c3),
            ChannelAttention(c3, c4)
        ])
        
        self.freq_spatial_attention = nn.ModuleList([
            SpatialAttention(scale_factor=2),
            SpatialAttention(scale_factor=4)
        ])
        
        self.rgb_final = nn.Conv2d(c1 + c2 + c3 + c4, 1, 1)
        self.freq_final = nn.Conv2d(c2 + c3 + c4, 1, 1)
        
    def _apply_attention(
        self, 
        features: List[torch.Tensor],
        channel_attention: nn.ModuleList,
        spatial_attention: nn.ModuleList
    ) -> List[torch.Tensor]:
        
        for i in range(len(features) - 2, -1, -1):
            features[i] = channel_attention[i](features[i], features[i+1])
                    
        for i in range(1, len(features)):
            features[i] = spatial_attention[i-1](features[i-1], features[i])
            
        return features
    
    def _fuse_predictions(self, rgb_pred: torch.Tensor, freq_pred: torch.Tensor) -> torch.Tensor:
        if freq_pred.shape != rgb_pred.shape:
            freq_pred = F.interpolate(
                freq_pred, 
                size=rgb_pred.shape[2:], 
                mode='bilinear',
                align_corners=False
            )

                
        predictions = torch.cat([rgb_pred, freq_pred], dim=1)
        weights = F.softmax(predictions, dim=1)
        
        main_weights = torch.max(weights, dim=2, keepdim=True)[0]
        main_weights = torch.max(main_weights, dim=3, keepdim=True)[0]
        
        final_pred = (
            main_weights[:, 0:1] * rgb_pred + 
            main_weights[:, 1:2] * freq_pred
        )
         
        return final_pred
        
    def forward(
        self,
        rgb_input: torch.Tensor,
        dct_coeffs: Optional[torch.Tensor] = None,
        q_matrix: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            rgb_input: RGB image input (B, 3, H, W)
            dct_coeffs: DCT coefficients (B, H, W) or None
            q_matrix: Quantization matrix (B, H, W) or None
            mask: Optional mask (B, H, W)
        Returns:
            Final prediction mask (B, 1, H, W)
        """
        
        rgb_feats = self.rgb_hrnet(rgb_input)
        
        a, b, c, d = rgb_feats[0], rgb_feats[1], rgb_feats[2], rgb_feats[3]
        rgb_feats = [a, b, c, d]
        
        
        rgb_feats = [aspp(feat) for aspp, feat in zip(self.rgb_aspp, rgb_feats)]

        rgb_feats = self._apply_attention(
            rgb_feats,
            self.rgb_channel_attention,
            self.rgb_spatial_attention
        )
        
        rgb_concat = torch.cat(rgb_feats, dim=1)
        rgb_pred = self.rgb_final(rgb_concat)
        
        final_pred = rgb_pred
        
        final_pred = F.interpolate(
            final_pred,
            size=rgb_input.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return torch.sigmoid(final_pred)

if __name__ == "__main__":
    model = TwoBranchNetwork()
    
    batch_size = 2
    height, width = 256, 256  
    
    rgb_input = torch.randn(batch_size, 3, height, width)
    dct_coeffs = torch.randn(batch_size, height, width)
    q_matrix = torch.ones(batch_size, height, width)
    mask = torch.ones(batch_size, height, width)
    
    # Forward pass
    output = model(rgb_input, dct_coeffs, q_matrix, mask)
    print(f"Output shape: {output.shape}")
