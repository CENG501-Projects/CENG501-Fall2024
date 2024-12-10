import torch
import torch.nn as nn

from modules.LocalFeatureExtraction import UNetFeatureExtractor
from modules.ViewSwitching import ViewSwitcher
from modules.ConflictFreeCoarseMatching import ConflictFreeCoarseMatchingModule
from modules.FineMatching import FineMatching


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
        self.fine_matching = FineMatching()
    
    def forward(self, img1, img2):
        # 1. Feature Extraction for 2 images
        features1 = self.unet(img1)  # img1
        features2 = self.unet(img2)  # img2
        
        # 2. View Switching
        #switched_features1, switched_features2 = self.view_switcher(features1, features2)
        
        # 3. Conflict-Free Coarse Matching
        #coarse_matches = self.coarse_matching(switched_features1, switched_features2)
        
        # 4.  Fine Matching
        #fine_matches = self.fine_matching(coarse_matches)
        
        #return fine_matches
