def trilinear_interpolate(point, corners):
    """
    Performs trilinear interpolation for a query point among 8 corner voxels.

    Args:
        point (tuple or list of 3 floats): (px, py, pz) - the query point
        corners (list of 8 dicts):
            Each dict has:
              {
                'pos': (x, y, z),  # corner coords
                'occ': float,      # occupancy/depth
                'col': (r, g, b)   # color
              }

    Returns:
        occ_val (float): interpolated occupancy/depth
        (r_val, g_val, b_val) (tuple of floats): interpolated color
    """
    px, py, pz = point

    # 1) Extract corner positions in a known ordering
    #    For example, let's assume corners[0] is (x0, y0, z0),
    #    corners[1] is (x0, y0, z1),
    #    ...
    #    corners[7] is (x1, y1, z1).
    # If your corners are not in that order, you need to reorder them or do a small sorting step.

    # We assume:
    # corners[0] => (x0, y0, z0)
    # corners[1] => (x0, y0, z1)
    # corners[2] => (x0, y1, z0)
    # corners[3] => (x0, y1, z1)
    # corners[4] => (x1, y0, z0)
    # corners[5] => (x1, y0, z1)
    # corners[6] => (x1, y1, z0)
    # corners[7] => (x1, y1, z1)

    # For brevity, just do direct indexing:
    pos0 = corners[0]['pos']  # (x0, y0, z0)
    pos7 = corners[7]['pos']  # (x1, y1, z1)

    x0, y0, z0 = pos0
    x1, y1, z1 = pos7

    # 2) Compute fractional offsets in each dimension
    #    fraction in x: dx = (px - x0) / (x1 - x0)
    #    fraction in y: dy = (py - y0) / (y1 - y0)
    #    fraction in z: dz = (pz - z0) / (z1 - z0)
    # Add small checks to avoid divide-by-zero if corners degenerate.
    denom_x = (x1 - x0) if (x1 != x0) else 1e-6
    denom_y = (y1 - y0) if (y1 != y0) else 1e-6
    denom_z = (z1 - z0) if (z1 != z0) else 1e-6

    dx = (px - x0) / denom_x
    dy = (py - y0) / denom_y
    dz = (pz - z0) / denom_z

    # 3) Gather corner occupancy & color
    #    We'll label them val_000, val_001, ..., val_111
    val_000 = corners[0]['occ']  # (x0, y0, z0)
    val_001 = corners[1]['occ']  # (x0, y0, z1)
    val_010 = corners[2]['occ']  # (x0, y1, z0)
    val_011 = corners[3]['occ']  # (x0, y1, z1)
    val_100 = corners[4]['occ']  # (x1, y0, z0)
    val_101 = corners[5]['occ']  # (x1, y0, z1)
    val_110 = corners[6]['occ']  # (x1, y1, z0)
    val_111 = corners[7]['occ']  # (x1, y1, z1)

    # 4) Interpolate occupancy
    #    standard trilinear weighting:
    occ_val = (
        val_000 * (1-dx)*(1-dy)*(1-dz) +
        val_100 * (dx)*(1-dy)*(1-dz)  +
        val_010 * (1-dx)*dy*(1-dz)    +
        val_110 * dx*dy*(1-dz)        +
        val_001 * (1-dx)*(1-dy)*dz    +
        val_101 * dx*(1-dy)*dz        +
        val_011 * (1-dx)*dy*dz        +
        val_111 * dx*dy*dz
    )

    # 5) Interpolate color similarly
    #    We do the same weighting for each channel (R, G, B)
    col_000 = corners[0]['col']  # (r0, g0, b0)
    col_001 = corners[1]['col']
    col_010 = corners[2]['col']
    col_011 = corners[3]['col']
    col_100 = corners[4]['col']
    col_101 = corners[5]['col']
    col_110 = corners[6]['col']
    col_111 = corners[7]['col']

    # We'll do the weighted sum for each channel separately
    def _interp_color(corner_color, w):
        # corner_color => (r_i, g_i, b_i), w => weight
        return (corner_color[0]*w, corner_color[1]*w, corner_color[2]*w)

    # We can accumulate in (r_sum, g_sum, b_sum)
    w_000 = (1-dx)*(1-dy)*(1-dz)
    w_100 = dx*(1-dy)*(1-dz)
    w_010 = (1-dx)*dy*(1-dz)
    w_110 = dx*dy*(1-dz)
    w_001 = (1-dx)*(1-dy)*dz
    w_101 = dx*(1-dy)*dz
    w_011 = (1-dx)*dy*dz
    w_111 = dx*dy*dz

    # r_sum, g_sum, b_sum
    r_sum = (col_000[0]*w_000 + col_100[0]*w_100 + col_010[0]*w_010 + col_110[0]*w_110 +
             col_001[0]*w_001 + col_101[0]*w_101 + col_011[0]*w_011 + col_111[0]*w_111)
    g_sum = (col_000[1]*w_000 + col_100[1]*w_100 + col_010[1]*w_010 + col_110[1]*w_110 +
             col_001[1]*w_001 + col_101[1]*w_101 + col_011[1]*w_011 + col_111[1]*w_111)
    b_sum = (col_000[2]*w_000 + col_100[2]*w_100 + col_010[2]*w_010 + col_110[2]*w_110 +
             col_001[2]*w_001 + col_101[2]*w_101 + col_011[2]*w_011 + col_111[2]*w_111)

    col_val = (r_sum, g_sum, b_sum)

    return occ_val, col_val
