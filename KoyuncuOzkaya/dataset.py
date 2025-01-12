# FILE: dataset.py
import torch
from torch.utils.data import Dataset
import random
import math

class RaySampledDataset(Dataset):
    """
    Each sample is one or more points/rays from the octomap. We retrieve
    ground truth occupancy (or depth) + color from the octomap's raycast.
    """
    def __init__(self, octomap, max_depth,origin=(0,0,0), max_range=5.0, dataset_size=1000):
        super().__init__()
        self.octomap = octomap
        self.origin = origin
        self.max_range = max_range
        self.dataset_size = dataset_size
        self.max_depth = max_depth

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # 1) random direction
        dx = random.uniform(-1,1)
        dy = random.uniform(-1,1)
        dz = random.uniform(0.1,1)
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        dx, dy, dz = dx/length, dy/length, dz/length
        direction = [dx, dy, dz]

        # 2) castRay -> get distance & color
        i = 0
        hit, dist, r,g,b = self.octomap.castRay(self.origin, direction, self.max_range)
        
        while not hit:
            i = i+1
            dx = random.uniform(-1,1)
            dy = random.uniform(-1,1)
            dz = random.uniform(0.1,1)
            length = math.sqrt(dx*dx + dy*dy + dz*dz)
            dx, dy, dz = dx/length, dy/length, dz/length
            direction = [dx, dy, dz]
            hit, dist, r,g,b = self.octomap.castRay(self.origin, direction, self.max_range)

        #if not hit:
        #    dist = 0.0  # or treat as free?
        #    r = g = b = 0

        # Suppose occupancy is 1 if hit, else 0:
        occ_gt = dist/self.max_depth
        color_gt = [r/255.0, g/255.0, b/255.0]

        # Convert to a 3D point in world coords:
        # If we want the actual 3D position of the hit, we can do:
        px = self.origin[0] + dist*dx
        py = self.origin[1] + dist*dy
        pz = self.origin[2] + dist*dz
        p_world = [px, py, pz]
        sample = {
            'p_world': torch.tensor(p_world, dtype=torch.float32),   # [3]
            'occ_gt': torch.tensor([occ_gt], dtype=torch.float32),   # [1]
            'color_gt': torch.tensor(color_gt, dtype=torch.float32), # [3]
        }
        return sample
