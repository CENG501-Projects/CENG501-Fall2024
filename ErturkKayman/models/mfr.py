import torch
import torch.nn as nn

class SimplifiedMFR(nn.Module):
    def __init__(self, token_dim, out_channels, map_size=(7, 7)):
        super().__init__()
        self.map_size = map_size  # H and W of the feature map to reconstruct
        self.token_dim = token_dim
        self.out_channels = out_channels

        # Upsampling to the original feature map size
        self.fc = nn.Linear(token_dim, map_size[0] * map_size[1] * out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, tokens):
        """
        Args:
            tokens: Tensor of shape (B, num_clusters, token_dim)
        Returns:
            feature_map: Tensor of shape (B, out_channels, H, W)
        """
        B, num_clusters, _ = tokens.shape

        # Fully connect tokens to create a rough feature map
        dense = self.fc(tokens)  # Shape: (B, num_clusters, H*W*out_channels)
        dense = dense.view(B, num_clusters, self.out_channels, self.map_size[0], self.map_size[1])
        dense = dense.mean(dim=1)  # Aggregate over clusters -> Shape: (B, out_channels, H, W)

        # Apply a convolution to refine the feature map
        feature_map = self.conv(dense)
        return feature_map

