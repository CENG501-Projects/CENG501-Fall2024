import torch.nn as nn
from models.backbone import Backbone
from models.att import SimplifiedATT
from models.mfr import SimplifiedMFR

class MonoATT(nn.Module):
    def __init__(self, num_clusters=16, token_dim=512, out_channels=256, map_size=(7, 7)):
        super().__init__()
        self.backbone = Backbone(pretrained=False)
        self.att = SimplifiedATT(num_clusters=num_clusters, embed_dim=token_dim)
        self.mfr = SimplifiedMFR(token_dim=token_dim, out_channels=out_channels, map_size=map_size)

        # Mono3D detection head (GUPNet or similar)
        self.detection_head = nn.Sequential(
            nn.Conv2d(out_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 10, kernel_size=1),  # 10 channels for 3D box params
        )

    def forward(self, x):
        """
        Args:
            x: Input image tensor (B, 3, H, W)
        Returns:
            output: Predicted 3D bounding box parameters
        """
        features = self.backbone(x)          # Extract features
        refined_tokens = self.att(features) # Apply ATT
        reconstructed_map = self.mfr(refined_tokens) # Reconstruct feature map
        output = self.detection_head(reconstructed_map) # Predict 3D bounding boxes
        return output

