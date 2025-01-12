import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusterCenterEstimation(nn.Module):
    def __init__(self, alpha = 1, num_clusters=100):
        """
        Initialize the Cluster Center Estimation module.
        Args:
            alpha (float): Weight for semantic scoring.
            depth_min (float): Minimum depth value.
            depth_max (float): Maximum depth value.
            num_clusters (int): Number of cluster centers to select.
        """
        super().__init__()
        self.alpha = alpha  # Semantic weight factor
        self.num_clusters = num_clusters
        # Heatmap head for semantic scoring
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=True),  #I am forced to use 256 because of complexity.
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, bias=True),  # Output single-channel heatmap
        )
    #TODO: Try 1,1 and 2,2 grid size.
    def forward(self, features, calibs, B=500, original_image_height=375, token_grid_size = (4, 4)):
        """
        calib (torch.Tensor): Calibration matrices of shape [B, 3, 4].
                               Calibration matrix format:
                               [[fx, 0, cx, tx],
                                [0, fy, cy, ty],
                                [0,  0,  1, tz]]
        """
        #print(features.tensors.shape)
        #print(calibs.shape)
        BS, _, H, W = features.tensors.shape
        v = torch.arange(0, H, device=features.tensors.device).view(1, H, 1).expand(BS, H, W)  # Shape: [B, H, W]
        # Extract fy and cy from the calibration matrix
        fy = calibs[:, 1, 1].view(-1, 1, 1)  # fy (second row, second column) -> Shape: [B, 1, 1]
        cy = calibs[:, 1, 2].view(-1, 1, 1)  # cy (second row, third column)  -> Shape: [B, 1, 1]
        cy = (H * cy) / original_image_height #Original image size.
        """
        print("v (Vertical Pixel Coordinates):", v)
        print("fy (Vertical Focal Length):", fy)
        print("cy (Vertical Principal Point):", cy)
        print("H (Feature Map Height):", H)
        """
        # Compute depth scores based on the formula
        depth_scores = -F.relu(B * (v - cy) / (fy * H))  # Shape: [B, H, W]

        # Generate heatmap for semantic scoring
        heatmap = torch.sigmoid(self.heatmap_head(features.tensors))  # [B, 1, H, W]
        heatmap = heatmap.squeeze(1)  # Remove the singleton channel dimension, [B, H, W]
        # Compute combined scores
        combined_scores = depth_scores + self.alpha * heatmap  # Weighted combination

        #At this point, we have scores for each coord H,W
        # Calculate mean scores for tokens
        token_h, token_w = token_grid_size
        token_scores = []
        cluster_centers = []
        tokens = []
        token_positions = []
        for b in range(BS):
            batch_scores = combined_scores[b] # Shape: [H, W]
            batch_features = features.tensors[b] # Shape: [C, H, W]
            mean_scores = []
            batch_centers = []
            batch_tokens = []
            batch_positions = []
            for i in range(0, H, token_h):  # Loop over token rows
                for j in range(0, W, token_w):  # Loop over token columns
                    token_region_scores = batch_scores[i:i + token_h, j:j + token_w]  # Shape: [token_h, token_w]
                    token_region_features = batch_features[:, i:i + token_h, j:j + token_w]  # Shape: [C, token_h, token_w]

                    mean_score = token_region_scores.mean()  # Compute mean score
                    mean_features = token_region_features.mean(dim=(1, 2))  # Shape: [C]

                    # Append results
                    mean_scores.append(mean_score)
                    batch_centers.append(mean_features)

                    # Store token and its position
                    batch_tokens.append(mean_features)  # Shape: [C]
                    batch_positions.append(torch.tensor([i + token_h // 2, j + token_w // 2], device=features.tensors.device))  # Center position
     
            token_scores.append(torch.tensor(mean_scores, device=features.tensors.device))  # Shape: [num_tokens]
            cluster_centers.append(torch.stack(batch_centers))  # Shape: [num_tokens, C]
            tokens.append(torch.stack(batch_tokens))  # Shape: [num_tokens, C]
            token_positions.append(torch.stack(batch_positions))  # Shape: [num_tokens, 2]

        token_scores = torch.stack(token_scores)  # Shape: [B, num_tokens]
        cluster_centers = torch.stack(cluster_centers)  # Shape: [B, num_tokens, C]
        tokens = torch.stack(tokens)  # Shape: [B, num_tokens, C]
        token_positions = torch.stack(token_positions)  # Shape: [B, num_tokens, 2]

        _, top_indices = torch.topk(token_scores, self.num_clusters, dim=1)  # Indices of top tokens

        # Gather only top cluster center features
        final_cluster_centers = []
        for b in range(BS):
            top_centers = cluster_centers[b][top_indices[b]]  # Select top cluster centers for batch b
            final_cluster_centers.append(top_centers)
        final_cluster_centers = torch.stack(final_cluster_centers)  # Shape: [B, num_clusters, C]

        return combined_scores, final_cluster_centers, tokens, token_positions

    def normalize_heatmap(self, heatmap):
        """
        Normalize the heatmap to have values between 0 and 1.
        """
        return F.normalize(heatmap, p=2, dim=(-2, -1))

    def compute_depth_scores(self, depth_map):
        """
        Compute depth-based scores, prioritizing far or near objects.
        """
        depth_scores = -torch.log(depth_map + 1e-6)  # Example: Inverse depth
        return depth_scores

    def select_top_k(self, scores):
        """
        Select top-K cluster centers based on combined scores.
        Args:
            scores (torch.Tensor): Combined scores [B, H, W].
        Returns:
            cluster_centers (torch.Tensor): Top-K coordinates [B, K, 2].
        """
        B, H, W = scores.shape
        scores_flat = scores.view(B, -1)  # Flatten spatial dimensions
        _, topk_indices = torch.topk(scores_flat, self.num_clusters, dim=-1)

        # Convert flat indices to 2D coordinates
        y_coords = topk_indices // W
        x_coords = topk_indices % W
        cluster_centers = torch.stack((y_coords, x_coords), dim=-1)  # [B, K, 2]

        return cluster_centers


def build_cce(cfg):
    return ClusterCenterEstimation()