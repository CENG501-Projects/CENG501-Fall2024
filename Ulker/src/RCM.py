import torch
import torch.nn as nn

from src.modules.LocalFeatureExtraction import UNetFeatureExtractor
from src.modules.ViewSwitching import ViewSwitcher
from src.modules.ConflictFreeCoarseMatching import ConflictFreeCoarseMatchingModule
from src.modules.FineMatching import FineMatchingModule


# C1: Coarse features: 256
# C2: Fine features: 64
# L1: 5 layers of coarse attention 
# L2: 2 layers of fine attention
# N: Number of Keypoints 1024
# HxW: 832x832



class RCM(nn.Module):
    def __init__(self):
        super(RCM, self).__init__()
        self.params_c1 = 256
        self.params_c2 = 64
        
        # Sub-modules
        self.unet = UNetFeatureExtractor(in_channels=3, base_channels=self.params_c2)
        self.view_switcher = ViewSwitcher(threshold = 0.5)
        self.coarse_matching = ConflictFreeCoarseMatchingModule(self.params_c1)
        self.fine_matching = FineMatchingModule(self.params_c1)
    
    def forward(self, img1, img2):
        # 1. Feature Extraction for 2 images
        [f11, f12, f13, f14] = self.unet(img1)  # img1
        [f21, f22, f23, f24] = self.unet(img2)  # img2
        
        features1 = self.merge_features(f11, f12, f13, f14)
        features2 = self.merge_features(f21, f22, f23, f24)
        
        # 2. View Switching
        sparse_features, larger_scale_map, smaller_scale_map = self.view_switcher(features1, features2)
        
        # 3. Conflict-Free Coarse Matching
        coarse_matches, larger_scale_map, smaller_scale_map = self.coarse_matching(larger_scale_map, smaller_scale_map, sparse_features)
        # 4.  Fine Matching
        fine_matches = self.fine_matching(larger_scale_map.reshape(1, -1, 208, 208), smaller_scale_map.reshape(1, -1, 208, 208), coarse_matches.reshape(1, -1, 208, 208))
        
        return fine_matches

    def merge_features(self, f1, f2, f3, f4):
        # Tensor'leri yeniden boyutlandır (resize)
        f1_resized = f1.reshape((1, -1, 832, 832))
        f2_resized = f2.reshape((1, -1, 832, 832))
        f3_resized = f3.reshape((1, -1, 832, 832))

        # Birleştirme işlemi
        merged_tensor = torch.cat((f1_resized, f2_resized, f3_resized, f4), dim=1)

        return merged_tensor