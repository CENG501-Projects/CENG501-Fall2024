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
        
        # RGB HRNet stages
        self.rgb_hrnet = timm.create_model('hrnet_w18', pretrained=True)
        self.rgb_hrnet.eval()
        for param in self.rgb_hrnet.parameters():
            param.requires_grad = False 

        # Frequency Stream
        self.freq_stream = CompressionArtifactsLearning()
        
        # Frequency HRNet stages (3 resolutions as mentioned in paper)
        self.freq_stage1 = HRNetStage([96], [96])
        self.freq_stage2 = HRNetStage([96, 192], [96, 192])
        self.freq_stage3 = HRNetStage([96, 192, 384], [96, 192, 384])
        
        # ASPP modules for RGB stream
        self.rgb_aspp = nn.ModuleList([
            ASPP(c1, c1),
            ASPP(c2, c2),
            ASPP(c3, c3),
            ASPP(c4, c4)
        ])
        
        # ASPP modules for frequency stream
        self.freq_aspp = nn.ModuleList([
            ASPP(c2, c2),
            ASPP(c3, c3),
            ASPP(c4, c4)
        ])
        
        # Interactive attention modules
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
        
        # Final prediction layers
        self.rgb_final = nn.Conv2d(c1 + c2 + c3 + c4, 1, 1)
        self.freq_final = nn.Conv2d(c2 + c3 + c4, 1, 1)
        
    def _apply_attention(
        self, 
        features: List[torch.Tensor],
        channel_attention: nn.ModuleList,
        spatial_attention: nn.ModuleList
    ) -> List[torch.Tensor]:
        """Apply interactive attention mechanism"""
        # Channel attention (bottom-up)
        
        for i in range(len(features) - 2, -1, -1):
            features[i] = channel_attention[i](features[i], features[i+1])
                    
        # Spatial attention (top-down)
        for i in range(1, len(features)):
            features[i] = spatial_attention[i-1](features[i-1], features[i])
            
        return features
    
    def _fuse_predictions(self, rgb_pred: torch.Tensor, freq_pred: torch.Tensor) -> torch.Tensor:
        """Adaptive weighted heatmap aggregation"""
        # Ensure same spatial dimensions
        if freq_pred.shape != rgb_pred.shape:
            freq_pred = F.interpolate(
                freq_pred, 
                size=rgb_pred.shape[2:], 
                mode='bilinear',
                align_corners=False
            )

                
        predictions = torch.cat([rgb_pred, freq_pred], dim=1)
        weights = F.softmax(predictions, dim=1)
        
        # Global max pooling for weight selection
        main_weights = torch.max(weights, dim=2, keepdim=True)[0]
        main_weights = torch.max(main_weights, dim=3, keepdim=True)[0]
        
        # Weighted combination
        final_pred = (
            main_weights[:, 0:1] * rgb_pred + 
            main_weights[:, 1:2] * freq_pred
        )
         
        # Apply softmax
        # rgb_reshape = rgb_pred.view(rgb_pred.size(0), -1)
        # rgb_softmax = F.softmax(rgb_reshape, dim=1).view_as(rgb_pred)
        
        # freq_reshape = freq_pred.view(freq_pred.size(0), -1)
        # freq_softmax = F.softmax(freq_reshape, dim=1).view_as(freq_pred)

        # print(f"RGB softmax shape: {rgb_softmax.shape}")
        # print(f"Freq softmax shape: {freq_softmax.shape}")
        
        # print('rgb_softmax')
        # print (rgb_softmax)
        
        
        # rgb_max = rgb_softmax.max(dim=2)[1]
        # print(rgb_max)
        # rgb_max = rgb_max.max(dim=2)[0]

        # freq_max = freq_softmax.max(dim=2)[0]
        # freq_max = freq_max.max(dim=2)[0]
        
        # print(f"RGB max shape: {rgb_max.size()}")
        # print(f"Freq max shape: {freq_max.size()}")
        
        # # Global max pooling for weight selection
        # main_weights = torch.max(weights, dim=2, keepdim=True)[0]
        # main_weights = torch.max(main_weights, dim=3, keepdim=True)[0]
        
        # # Weighted combination
        # final_pred = (
        #     main_weights[:, 0:1] * rgb_pred + 
        #     main_weights[:, 1:2] * freq_pred
        # )
        
        return final_pred
        
    def forward(
        self,
        rgb_input: torch.Tensor,
        dct_coeffs: Optional[torch.Tensor] = None,
        q_matrix: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the two-branch network
        Args:
            rgb_input: RGB image input (B, 3, H, W)
            dct_coeffs: DCT coefficients (B, H, W) or None
            q_matrix: Quantization matrix (B, H, W) or None
            mask: Optional mask (B, H, W)
        Returns:
            Final prediction mask (B, 1, H, W)
        """
        # RGB Stream
        
        # HRNet stages for RGB
        rgb_feats = self.rgb_hrnet(rgb_input)
        
        a, b, c, d = rgb_feats[0], rgb_feats[1], rgb_feats[2], rgb_feats[3]
        rgb_feats = [a, b, c, d]
        
        
        # Apply ASPP to RGB features
        rgb_feats = [aspp(feat) for aspp, feat in zip(self.rgb_aspp, rgb_feats)]

        # Apply interactive attention to RGB features
        rgb_feats = self._apply_attention(
            rgb_feats,
            self.rgb_channel_attention,
            self.rgb_spatial_attention
        )
        
        # Frequency Stream (if inputs provided)
        if dct_coeffs is not None and q_matrix is not None:
            # Get initial frequency features
            freq_feat = self.freq_stream(dct_coeffs, q_matrix, mask)
            freq_feats = [freq_feat]
            
            # HRNet stages for frequency
            """freq_feats = self.freq_stage1(freq_feats)
            freq_feats = self.freq_stage2(freq_feats)
            freq_feats = self.freq_stage3(freq_feats)"""

            freq_feats = []
            # freq_feats.append(torch.rand(2,96,32,32))
            # freq_feats.append(torch.rand(2,192,16,16))
            # freq_feats.append(torch.rand(2,384,8,8))
            
            # Apply ASPP to frequency features
            freq_feats = [aspp(feat) for aspp, feat in zip(self.freq_aspp, freq_feats)]
            
            # Apply interactive attention to frequency features
            freq_feats = self._apply_attention(
                freq_feats,
                self.freq_channel_attention,
                self.freq_spatial_attention
            )
            
            # Generate frequency prediction
            freq_concat = torch.cat(freq_feats, dim=1)
            freq_concat = nn.Upsample(scale_factor=2, mode='bilinear')(freq_concat)
            freq_pred = self.freq_final(freq_concat)
        else:
            freq_pred = None
            
        # Generate RGB prediction
        rgb_concat = torch.cat(rgb_feats, dim=1)
        rgb_pred = self.rgb_final(rgb_concat)
        
        # Final prediction
        if freq_pred is not None:
            final_pred = self._fuse_predictions(rgb_pred, freq_pred)
        else:
            final_pred = rgb_pred
        
        # Resize to input resolution
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
    
    output = model(rgb_input, dct_coeffs, q_matrix, mask)
    print(f"Output shape: {output.shape}")
