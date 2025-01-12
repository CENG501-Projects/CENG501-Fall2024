import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class PositionEncoder(nn.Module):
    def __init__(self, feature_dim):
        super(PositionEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            # 
            nn.Linear(64, feature_dim)
        )


    def forward(self, features, positions):
        # Get last layer that shows no keypoint as 1
        positions = 1 - positions[0,64,:,:]
        # then apply mlp
        tensor_reshaped = positions.view(-1, 1)  # Shape will be (832*832, 1)
        # Apply MLP to each cell of the tensor (each element in the reshaped tensor)
        output_reshaped = self.mlp(tensor_reshaped)
        # Reshape back to the original tensor shape
        positions = output_reshaped.view(1, 3, 832, 832)  # Shape will be (3, 832, 832)
        return torch.cat((features, positions), dim=1)

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, query, key, value):
        attention_output, _ = self.attention(query, key, value)
        return attention_output + self.feed_forward(attention_output)

class ManyToOneMatching(nn.Module):
    def __init__(self, feature_dim, temperature=0.1):
        super(ManyToOneMatching, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.dustbin = nn.Parameter(torch.randn(1, feature_dim, feature_dim))

    def forward(self, sparse_features, dense_features):
        dense_with_dustbin = torch.cat([dense_features, self.dustbin], dim=0)
        scores = torch.matmul(sparse_features, dense_features) / self.temperature
        probabilities = F.softmax(scores, dim=-1)
        return probabilities

class ConflictFreeCoarseMatchingModule(nn.Module):
    def __init__(self, feature_dim):
        super(ConflictFreeCoarseMatchingModule, self).__init__()
        self.position_encoder = PositionEncoder(feature_dim=3)
        self.self_attention = AttentionLayer(embed_dim=208)
        self.cross_attention = AttentionLayer(embed_dim=208)
        self.matching_layer = ManyToOneMatching(feature_dim=208)
        self.pool = torch.nn.MaxPool2d(kernel_size=4, stride=4)

    def forward(self, sparse_features, dense_features, sparse_positions):
        # Encode positional information
        sparse_features = self.position_encoder(sparse_features, sparse_positions)

        # Self-attention for sparse and dense features
        sparse_features, dense_features = self.pool(sparse_features.reshape(-1, 832, 832)), self.pool(dense_features.reshape(-1, 832, 832))
        sparse_features = self.self_attention(sparse_features, sparse_features, sparse_features)
        dense_features = self.self_attention(dense_features, dense_features, dense_features)
        
        # Cross-attention between sparse and dense features
        sparse_features = self.cross_attention(sparse_features[0:90,:,:], dense_features, dense_features)
        
        # Many-to-one matching
        match_probabilities = self.matching_layer(sparse_features, dense_features)

        return match_probabilities, sparse_features, dense_features