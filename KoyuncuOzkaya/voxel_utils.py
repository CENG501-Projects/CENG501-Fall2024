import numpy as np
import torch

def build_list_of_voxel_grids(octo,
                              bbox_min=( -1.0, -1.0, 0.0 ),
                              bbox_max=(  1.0,  1.0, 2.0 ),
                              base_resolution=(32, 32, 32),
                              num_levels=2):
    """
    Builds multiple dense voxel grids from an OctoMap, for example:
      - level 0: 32x32x32
      - level 1: 64x64x64
    Each voxel stores [occupancy, R, G, B] or something similar.

    Args:
      octo (OctoMapWrapper): The octomap object.
      bbox_min (tuple): bounding box min in (x,y,z).
      bbox_max (tuple): bounding box max in (x,y,z).
      base_resolution (tuple): e.g. (32,32,32).
      num_levels (int): how many scales to create.

    Returns:
      list_of_grids: a list of PyTorch tensors,
         each shape [C_in, D_k, H_k, W_k],
         for example C_in=4 => (occupancy, R, G, B).
    """
    list_of_grids = []
    (x_min, y_min, z_min) = bbox_min
    (x_max, y_max, z_max) = bbox_max
    D0, H0, W0 = base_resolution

    # For demonstration, we treat these as "levels"
    for lvl in range(num_levels):
        D_k = D0*(2**lvl)
        H_k = H0*(2**lvl)
        W_k = W0*(2**lvl)

        # We'll store: occupancy + (r,g,b)
        C_in = 4
        grid_np = np.zeros((C_in, D_k, H_k, W_k), dtype=np.float32)

        for d in range(D_k):
            for h in range(H_k):
                for w_ in range(W_k):
                    # Map (d,h,w) -> world coords
                    xd = x_min + (x_max - x_min)*(d+0.5)/D_k
                    yd = y_min + (y_max - y_min)*(h+0.5)/H_k
                    zd = z_min + (z_max - z_min)*(w_+0.5)/W_k

                    # Attempt to 'search' or cast ray. If node is found:
                    # We'll do a quick "search" approach (no full ray-cast here):
                    node = None
                    # If you have a "search" function in your wrapper, it might be named differently:
                    # node = octo.search(xd, yd, zd)
                    # but if you only have castRay, you might do a short cast from above?
                    # For simplicity, we do a short cast from (xd, yd, zd) in some direction 
                    # or assume occupancy if castRay hits quickly. 
                    # We'll just do a hack: cast from above:
                    origin = [xd, yd, z_max + 0.1]
                    direction = [0, 0, -1]
                    hit, dist, r_, g_, b_ = octo.castRay(origin, direction, (z_max - z_min) + 1.0)

                    # If hit, check distance
                    # If the hit point is near (xd, yd, zd), assume it is occupied:
                    # (Of course, in a real scenario you might use the OctoMap search directly.)

                    occupancy = 0.0
                    rr = gg = bb = 0.0
                    if hit:
                        # We can see if (z_max+0.1 - dist) is near zd:
                        approx_z = (z_max+0.1) - dist
                        if abs(approx_z - zd) < 0.05:
                            occupancy = approx_z
                            rr, gg, bb = r_/255.0, g_/255.0, b_/255.0

                    grid_np[0, d,h,w_] = occupancy
                    grid_np[1, d,h,w_] = rr
                    grid_np[2, d,h,w_] = gg
                    grid_np[3, d,h,w_] = bb

        grid_torch = torch.from_numpy(grid_np)
        list_of_grids.append(grid_torch)

    return list_of_grids
