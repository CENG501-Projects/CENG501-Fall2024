import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveTokenClustering(nn.Module):
    def __init__(self, beta=1, token_dim=1024, num_clusters=100, d_k=256):
        super().__init__()
        self.beta = beta
        self.d_k = d_k
        self.num_clusters = num_clusters

        # Attention score generator for tokens
        self.attention_layer = nn.Linear(token_dim, 1)

        #Compulsory projection layer because of the computational complexity.
        d_model = 256
        self.projection = nn.Linear(token_dim, d_model)

        # Transformer for feature merging
        self.transformer = nn.Transformer(
            d_model=d_model, #Because of comp complexity inputs are projected.
            nhead=8, 
            num_encoder_layers=1,  #Because of the comp complexity 1 layer is used.
            num_decoder_layers=1, 
            dim_feedforward=512, 
            dropout=0.1
        )

    def forward(self, tokens, cluster_centers, positions):
        #Outline-preferred Token Grouping is implemented inside the forward dunction.
        B, num_tokens, C = tokens.shape
        _, num_clusters, _ = cluster_centers.shape
    
        # Initialize a tensor to store cluster assignments
        cluster_assignments = torch.zeros(B, num_tokens, dtype=torch.long, device=tokens.device)

        for b in range(B):
            batch_tokens = tokens[b]  # Shape: [num_tokens, C]
            batch_centers = cluster_centers[b]  # Shape: [num_clusters, C]
            batch_positions = positions[b]  # Shape: [num_tokens, 2]
            batch_positions = batch_positions.float()

            # Compute feature distances
            feature_distances = torch.cdist(batch_tokens, batch_centers, p=2)  # Shape: [num_tokens, num_clusters]

            # Compute spatial distances
            spatial_distances = torch.cdist(batch_positions, batch_positions, p=2)  # Shape: [num_tokens, num_tokens]
            spatial_distances = spatial_distances.mean(dim=1, keepdim=True)  # Reduce spatial distances

            # Calculate combined distances (δi)
            combined_distances = feature_distances - self.beta * spatial_distances  # Shape: [num_tokens, num_clusters]

            # Assign each token to the nearest cluster center
            cluster_assignments[b] = torch.argmin(combined_distances, dim=1)  # Shape: [num_tokens]

        merged_features = self.attentionBasedFeatureMerging(tokens, cluster_centers, cluster_assignments)
        """
        print("tokens: ", tokens.shape)
        print("cluster_centers: ", cluster_centers.shape)
        print("positions: ", positions.shape)
        print("merged_features: ", merged_features.shape)
        """
        return cluster_assignments, merged_features

    def attentionBasedFeatureMerging(self, tokens, cluster_centers, cluster_assignments):
        B, num_tokens, token_dim = tokens.shape
        # Step 1: Compute attention scores for tokens
        attention_scores = self.attention_layer(tokens).squeeze(-1)  # Shape: [B, num_tokens]
        attention_weights = F.softmax(attention_scores, dim=1)  # Normalize attention scores

        # Step 2: Prepare tokens and cluster centers for the transformer
        # Use cluster centers as queries, tokens as keys and values
        queries = cluster_centers.permute(1, 0, 2)  # Shape: [num_clusters, B, token_dim]
        keys = tokens.permute(1, 0, 2)  # Shape: [num_tokens, B, token_dim]
        values = tokens.permute(1, 0, 2)  # Shape: [num_tokens, B, token_dim]

        # Incorporate attention scores into keys/queries interaction
        token_attention_matrix = attention_weights.unsqueeze(1).expand(-1, self.num_clusters, attention_weights.size(1))  # [B, num_clusters, num_tokens]
        token_attention_matrix = token_attention_matrix.permute(2, 0, 1)  # Shape: [num_tokens, B, num_clusters]

        # Step 3: Pass through transformer
        # Add the token attention scores to transformer attention mechanism


        keys_reduced = self.projection(keys)
        queries_reduced = self.projection(queries)
        transformer_output = self.transformer(
            src=keys_reduced, tgt=queries_reduced, src_key_padding_mask=None, tgt_key_padding_mask=None
        )  # Shape: [num_clusters, B, token_dim]
        
        #transformer_output = torch.zeros(cluster_centers.shape[1], cluster_centers.shape[0], token_dim).to('cuda')
        # Reshape back to [B, num_clusters, token_dim]
        merged_features = transformer_output.permute(1, 0, 2)

        return merged_features

def build_atc(cfg):
    return AdaptiveTokenClustering()