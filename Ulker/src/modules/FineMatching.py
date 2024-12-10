import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.ConflictFreeCoarseMatching import AttentionLayer

class FineMatchingModule(nn.Module):
    def __init__(self, feature_dim, fine_window_size=5, attention_layers=2):
        super(FineMatchingModule, self).__init__()
        self.fine_window_size = fine_window_size
        self.cross_attention = nn.ModuleList([
            AttentionLayer(feature_dim) for _ in range(attention_layers)
        ])
        self.conv_integration = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)

    def crop_fine_features(self, fine_features, positions, window_size):
        """
        Fine features cropping based on coarse match positions.
        Args:
            fine_features: Tensor, shape (B, H, W, C)
            positions: Tensor, shape (B, K, 2), coarse match positions
            window_size: int, window size for target cropping
        Returns:
            source_crops: Tensor, shape (B, K, C)
            target_crops: Tensor, shape (B, K, window_size^2, C)
        """
        B, H, W, C = fine_features.shape
        K = positions.shape[1]
        fine_features = fine_features.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Extract patches for target image
        patches = F.unfold(fine_features, kernel_size=window_size, padding=window_size//2)
        patches = patches.permute(0, 2, 1).reshape(B, H, W, window_size**2, C)

        # Source features at exact positions
        positions = positions.long()
        source_crops = fine_features[torch.arange(B).unsqueeze(1), :, positions[:, :, 0], positions[:, :, 1]]

        # Target patches at corresponding positions
        target_crops = patches[torch.arange(B).unsqueeze(1), positions[:, :, 0], positions[:, :, 1]]

        return source_crops, target_crops

    def forward(self, source_features, target_features, coarse_positions):
        """
        Perform fine matching refinement.
        Args:
            source_features: Tensor, shape (B, H, W, C)
            target_features: Tensor, shape (B, H, W, C)
            coarse_positions: Tensor, shape (B, K, 2), coarse match positions
        Returns:
            refined_positions: Tensor, shape (B, K, 2), fine match positions
        """
        B, H, W, C = target_features.shape

        # Crop fine features based on coarse positions
        source_crops, target_crops = self.crop_fine_features(target_features, coarse_positions, self.fine_window_size)

        # Apply cross-attention
        for layer in self.cross_attention:
            target_crops = layer(source_crops, target_crops, target_crops)

        # Correlation map and refined positions
        correlation_map = torch.einsum('bkc,bknc->bkn', source_crops, target_crops)  # (B, K, window_size^2)
        probabilities = F.softmax(correlation_map, dim=-1)  # (B, K, window_size^2)

        # Compute refined positions using expectation (soft-argmax)
        offsets = torch.linspace(-self.fine_window_size//2, self.fine_window_size//2, self.fine_window_size).to(source_features.device)
        offsets_x, offsets_y = torch.meshgrid(offsets, offsets, indexing='ij')
        offsets = torch.stack([offsets_x.flatten(), offsets_y.flatten()], dim=-1)  # (window_size^2, 2)
        refined_offsets = torch.einsum('bkn,nd->bkd', probabilities, offsets)  # (B, K, 2)

        refined_positions = coarse_positions + refined_offsets

        return refined_positions
