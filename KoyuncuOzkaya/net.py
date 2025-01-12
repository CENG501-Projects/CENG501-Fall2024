import torch
import torch.nn as nn
import torch.nn.functional as F
from unet3d import UNet3D  # your 3D U-Net
from positional_encode import positional_encoding_sin_cos
from interpolate import trilinear_interpolate
import numpy as np

class HierarchicalEncoder(nn.Module):
    """
    1) For each level k, we have a 3D voxel grid: [C_in, D_k, H_k, W_k].
    2) We run UNet3D -> [c_dim, D_k, H_k, W_k].
    3) For a query point p in [0,1]^3 coords, do trilinear sampling -> [c_dim].
    4) Expand with alpha_k/beta_k, then max-pool across levels.
    5) Also do sinusoidal pos-enc of p (in camera or world coords).
    6) Return final voxel embedding + pos-enc as a single vector.
    """

    def __init__(
        self,
        list_of_voxel_grids, 
        c_dim=16,
        max_levels=5,
        alpha_beta_dim="scalar",
        num_pos_freqs=10,
        **unet3d_kwargs
    ):
        super().__init__()
        self.K = min(max_levels, len(list_of_voxel_grids))

        # store voxel grids as buffers or non-trainable params
        self.voxel_grids = nn.ParameterList([
            nn.Parameter(g, requires_grad=False) for g in list_of_voxel_grids
        ])

        # Build a UNet3D for each level
        self.unets = nn.ModuleList([
            UNet3D(
                in_channels=self.voxel_grids[k].shape[0],
                out_channels=c_dim,
                **unet3d_kwargs
            ) for k in range(self.K)
        ])

        # alpha_k, beta_k
        if alpha_beta_dim == "scalar":
            self.alpha = nn.ParameterList([
                nn.Parameter(torch.ones(1)) for _ in range(self.K)
            ])
            self.beta = nn.ParameterList([
                nn.Parameter(torch.ones(1)) for _ in range(self.K)
            ])
        else:
            # e.g. vector of length c_dim
            self.alpha = nn.ParameterList([
                nn.Parameter(torch.ones(c_dim)) for _ in range(self.K)
            ])
            self.beta = nn.ParameterList([
                nn.Parameter(torch.ones(c_dim)) for _ in range(self.K)
            ])

        self.c_dim = c_dim
        self.num_pos_freqs = num_pos_freqs

    def expand_feature(self, feat, alpha, expansion_factor):
        """
        feat: shape [B, feat_dim]
        alpha: shape [1] or [feat_dim]
        expansion_factor: integer
        => [B, feat_dim * (expansion_factor+1)]
        """
        expansions = []
        expansions.append(feat)  # alpha^0 * feat

        alpha_power = alpha
        for _ in range(expansion_factor):
            new_feat = feat * alpha_power
            expansions.append(new_feat)
            alpha_power = alpha_power * alpha

        return torch.cat(expansions, dim=-1)

    def forward(self, p_coords, p_world):
        """
        p_coords: [B, 3], query points in normalized coords for voxel sampling
                  (range [0,1]^3 for each level).
        p_world:  [B, 3], the same points but in world/camera coords if you want 
                  to do positional encoding. If None, we skip pos-enc or do something else.

        Return:
          final_embedding: [B, final_dim], where final_dim = voxel_feat_dim + pos_enc_dim
                           you can feed this to your decoder.
        """
        device = p_coords.device
        B = p_coords.size(0)

        # 1) pass each voxel grid -> UNet3D -> c_k
        c_feats = []
        for k in range(self.K):
            with torch.no_grad():
                grid_in = self.voxel_grids[k]  # [C_in, D_k, H_k, W_k]
            c_k = self.unets[k](grid_in.unsqueeze(0))  # => [1, c_dim, D_k, H_k, W_k]
            c_feats.append(c_k.squeeze(0))  # 

        # 2) for each scale k, do trilinear sampling
        all_depth_expanded = []
        all_color_expanded = []

        for k in range(self.K):
            c_k = c_feats[k].unsqueeze(0)  # => [1, c_dim, D_k, H_k, W_k]
            spatial_coords = p_coords[:, :3] 
            # build a vgrid => shape [B, 1, 1, 1, 3] => scale p_coords [0,1]->[-1,1]
            p_norm = 2.0 * spatial_coords - 1.0
            p_norm = p_norm.view(1, B, 1, 1, 3).float()

            sampled = F.grid_sample(
                c_k,
                p_norm,
                padding_mode='border',
                align_corners=True,
                mode='bilinear'
            )


            sampled = sampled.squeeze(-1).squeeze(-1)  # => [1, c_dim, B]
            sampled = sampled.squeeze(0).permute(1, 0)  # => [B, c_dim]

            # Suppose half is "depth" half is "color", or do it as you wish
            half_dim = self.c_dim // 2
            phi_d_k = sampled[:, :half_dim]  # => [B, half_dim]
            phi_c_k = sampled[:, half_dim:]  # => [B, half_dim]

            # expansions alpha/beta
            expansion_factor = 2 ** (self.K - 1 - k)
            phi_d_exp = self.expand_feature(phi_d_k, self.alpha[k], expansion_factor)
            phi_c_exp = self.expand_feature(phi_c_k, self.beta[k], expansion_factor)

            all_depth_expanded.append(phi_d_exp)
            all_color_expanded.append(phi_c_exp)
            #print(np.array(all_depth_expanded.cpu()).shape)

        depth_stack = torch.stack(all_depth_expanded, dim=0)  # => [K, B, ...]
        color_stack = torch.stack(all_color_expanded, dim=0)  # => [K, B, ...]

        phi_depth = depth_stack.abs().max(dim=0)[0]  # => [B, ...]
        phi_color = color_stack.abs().max(dim=0)[0]  # => [B, ...]

        phi_voxel = torch.cat([phi_depth, phi_color], dim=-1)  # => [B, voxel_feat_dim]

        # 3) do positional encoding if p_world is provided
        if p_world is not None:
            
            pos_enc = positional_encoding_sin_cos(p_world, num_freqs=self.num_pos_freqs)
            final_embedding = torch.cat([phi_voxel, pos_enc], dim=-1)  # => [B, voxel_feat_dim + pos_enc_dim]
        else:
            final_embedding = phi_voxel

        return final_embedding


    
class LocalPointDecoder(nn.Module):
    def __init__(self, latent_dim, pos_enc_dim, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + pos_enc_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        self.fc_depth = nn.Linear(hidden_size, 1)  # or occupancy
        self.fc_rgb   = nn.Linear(hidden_size, 3)

    def forward(self, final_latent, pos_enc):
        """
        final_latent: [B, latent_dim]
        pos_enc: [B, pos_enc_dim]
        """
        x = torch.cat([final_latent, pos_enc], dim=-1)  # => [B, latent_dim + pos_enc_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        depth_pred = self.fc_depth(x)  # => [B,1]
        rgb_pred   = self.fc_rgb(x)    # => [B,3]
        return depth_pred, rgb_pred
