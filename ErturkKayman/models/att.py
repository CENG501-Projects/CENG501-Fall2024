import torch
import torch.nn as nn
from sklearn.cluster import KMeans

class SimplifiedATT(nn.Module):
    def __init__(self, num_clusters=16, embed_dim=512):
        super().__init__()
        self.num_clusters = num_clusters
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)

    def forward(self, feature_map):
        """
        Args:
            feature_map: Tensor of shape (B, C, H, W)
        Returns:
            refined_tokens: Tensor of shape (B, num_clusters, embed_dim)
        """
        B, C, H, W = feature_map.shape
        # Flatten the feature map into tokens
        tokens = feature_map.view(B, C, -1).permute(0, 2, 1)  # Shape: (B, H*W, C)

        # Use KMeans clustering on the first batch to find cluster centers
        kmeans = KMeans(n_clusters=self.num_clusters)
        cluster_labels = kmeans.fit_predict(tokens[0].cpu().detach().numpy())  # (H*W)

        # Aggregate token features by cluster
        cluster_tokens = torch.zeros((B, self.num_clusters, C), device=feature_map.device)
        for cluster_id in range(self.num_clusters):
            mask = (torch.tensor(cluster_labels, device=feature_map.device) == cluster_id).float()
            mask = mask.unsqueeze(0).repeat(B, 1)  # Shape: (B, H*W)
            cluster_tokens[:, cluster_id, :] = (tokens * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)

        # Apply attention for refinement
        refined_tokens, _ = self.attention(cluster_tokens, cluster_tokens, cluster_tokens)
        return refined_tokens

