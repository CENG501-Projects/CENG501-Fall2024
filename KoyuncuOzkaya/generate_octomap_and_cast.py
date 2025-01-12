import cv2
import numpy as np
import build.octomap_pybind as opyb
import math
from torch.utils.data import DataLoader
import torch

from trainer import train
from dataset import RaySampledDataset
from net import HierarchicalEncoder, LocalPointDecoder
from voxel_utils import build_list_of_voxel_grids



def depth_to_color_points(depth_img: np.ndarray,
                          color_img: np.ndarray,
                          K: np.ndarray,
                          depth_scale: float = 1.0 / 1000.0):
    """
    Converts a depth map + color map into a list of ColoredPoint objects.
    
    Args:
      depth_img (np.ndarray): HxW float or uint16 (depth).
      color_img (np.ndarray): HxW x 3 (BGR).
      K (np.ndarray): 3x3 camera intrinsics.
      depth_scale (float): factor to convert raw depth to meters, if needed.

    Returns:
      A list of opyb.ColoredPoint objects (x,y,z,r,g,b).
    """
    points = []
    height, width = depth_img.shape[:2]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    for y in range(height):
        for x in range(width):
            d = float(depth_img[y, x]) * depth_scale  # convert to meters
            if d > 0:
                # Reproject pixel (x, y) -> 3D (camera coords)
                px = (x - cx) * d / fx
                py = (y - cy) * d / fy
                pz = d

                # BGR -> (b, g, r)
                b, g, r = color_img[y, x]

                # Build pybind struct
                cp = opyb.ColoredPoint()
                cp.x, cp.y, cp.z = px, py, pz
                cp.r, cp.g, cp.b = r, g, b  # store as 0..255
                points.append(cp)
    
    return points, np.max(depth_img)*depth_scale

def build_octomap(depth_path: str,
                  color_path: str,
                  K: np.ndarray,
                  resolution: float = 0.01,
                  depth_scale: float = 1.0 / 1000.0) -> opyb.OctoMapWrapper:
    """
    Builds and returns an OctoMap from the given depth/color images.

    Args:
      depth_path (str): Path to depth image.
      color_path (str): Path to color (RGB) image.
      K (np.ndarray): 3x3 camera intrinsics.
      resolution (float): Voxel resolution in meters (e.g. 0.01).
      depth_scale (float): Factor to convert raw depth to meters.

    Returns:
      opyb.OctoMapWrapper: The constructed ColorOcTree wrapper.
    """
    # 1) Load depth + color
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise FileNotFoundError(f"Could not load depth image at '{depth_path}'")

    color_img = cv2.imread(color_path, cv2.IMREAD_COLOR)
    if color_img is None:
        raise FileNotFoundError(f"Could not load color image at '{color_path}'")

    # Convert depth to float if necessary
    if depth_img.dtype != np.float32:
        depth_img = depth_img.astype(np.float32)

    # 2) Convert to colored 3D points
    points_list, max_depth = depth_to_color_points(depth_img, color_img, K, depth_scale)
    print(f"Converted {len(points_list)} points from depth+RGB")

    # 3) Build the OctoMap
    octo = opyb.OctoMapWrapper(resolution)
    octo.buildMap(points_list)
    octo.prune()

    # 4) Optional: save to .ot file
    # octo.save("map.ot")
    # print("OctoMap saved to map.ot")

    return octo, max_depth

def cast_ray(octo: opyb.OctoMapWrapper,
             origin: np.ndarray,
             direction: np.ndarray,
             max_range: float = 5.0):
    """
    Casts a ray in the given OctoMap, returns (hit, distance, color).

    Args:
      octo (opyb.OctoMapWrapper): The previously built OctoMap.
      origin (np.ndarray): shape [3], 3D origin of the ray.
      direction (np.ndarray): shape [3], direction vector (needn't be normalized).
      max_range (float): max search distance in meters.

    Returns:
      (bool, float, tuple): (hit, distance, (r,g,b))
        If hit=False, distance=-1, color=(0,0,0).
    """
    # Normalize direction
    d_len = np.linalg.norm(direction)
    if d_len > 1e-8:
        direction = direction / d_len

    # castRay returns: (bool hit, float dist, unsigned char r, g, b)
    hit, dist, rr, gg, bb = octo.castRay(origin.tolist(), direction.tolist(), max_range)
    return hit, dist, (rr, gg, bb)


def main():
    K = np.array([
        [721.5377,   0.0,    596.5593],
        [  0.0,   721.5377, 149.854 ],
        [  0.0,     0.0,      1.0   ]
    ], dtype=np.float32)

    depth_path = "deneme/depth.png"
    color_path = "deneme/rgb.png"

    # 1) Build the octomap once
    octo, max_depth = build_octomap(depth_path, color_path, K,
                         resolution=0.01, depth_scale=1.0/1000.0)

    grids = build_list_of_voxel_grids(octo, 
                                      bbox_min=(-1,-1,0),
                                      bbox_max=(1,1,2),
                                      base_resolution=(32,32,32), 
                                      num_levels=1)
    print("Built voxel grids:", [g.shape for g in grids])
    # 3) Create dataset for random rays => ground truth
    dataset = RaySampledDataset(
        octo,max_depth, origin=(0,0,0), max_range=300.0, dataset_size=5000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    valdataset = RaySampledDataset(
        octo,max_depth, origin=(0,0,0), max_range=300.0, dataset_size=1000)
    valloader = DataLoader(valdataset, batch_size=32, shuffle=True)

    # 3) Initialize the hierarchical encoder + decoder
    encoder = HierarchicalEncoder(list_of_voxel_grids=grids ,max_levels=1)
    decoder = LocalPointDecoder(latent_dim=32, pos_enc_dim=66, hidden_size=128)
    # We need to know what in_dim is. In your snippet, it's (phi_combined + p_coords).
    # phi_combined dimension = ?
    #   E.g. If each scale yields expansions => total final dimension = something
    #   Let's guess 24. Then plus coords=3 => in_dim=27, etc.

    # For now, let's do a dummy pass to figure out dimension:
    #p_dummy = torch.rand(1,3)
    #phi_dummy = encoder(p_dummy)  # shape [1, feat_dim], suppose feat_dim=16
    #in_dim = phi_dummy.shape[-1] + 3  # e.g. 16 + 3 = 19
    #print("Detected in_dim:", in_dim)

    # Re-init the decoder with correct in_dim
    #decoder = LocalPointDecoder(latent_dim=in_dim,pos_enc_dim=32,hidden_size=128)

    # 4) Train
    train(encoder, decoder, dataloader, valloader, device='cuda', num_epochs=16)

if __name__ == "__main__":
    main()
