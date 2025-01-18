import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.utils import padding

############################################################
#################### Created by ChatGPT ####################
############################################################
def warp_frame(image, flow):
    """
    Warps an image using the given optical flow.

    Args:
        image (torch.Tensor): Input image tensor of shape (B, 1, H, W).
        flow (torch.Tensor): Flow field tensor of shape (B, 2, H, W).

    Returns:
        torch.Tensor: Warped image of shape (B, 1, H, W).
    """
    # Get the shape of the image
    B, C, H, W = image.size()

    # Generate mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)  # Shape: (B, 1, H, W)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)  # Shape: (B, 1, H, W)

    # Stack xx and yy to create a grid
    grid = torch.cat((xx, yy), 1).float()  # Shape: (B, 2, H, W)

    # Move grid to the same device as the image
    if image.is_cuda:
        grid = grid.cuda()

    # Add flow field to the grid
    vgrid = grid + flow  # Warped grid coordinates

    # Normalize the grid to [-1, 1] range for grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / (W - 1) - 1.0  # X-axis
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / (H - 1) - 1.0  # Y-axis

    # Reshape grid to (B, H, W, 2) for grid_sample
    vgrid = vgrid.permute(0, 2, 3, 1)

    # Perform grid sampling
    warped_image = F.grid_sample(image, vgrid, mode='bilinear', padding_mode='zeros', align_corners=False)

    return warped_image

####################################################################################
############################ Adopted from Spike-FlowNet ############################
####################################################################################
def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    """
    Robust Charbonnier loss.
    """
    loss = torch.sum(torch.pow(torch.mul(delta,delta) + torch.mul(epsilon,epsilon), alpha))
    return loss
####################################################################################


def get_scaled_frame(frame, scale):
    if scale == 1.0:
        scaled_frame = frame  # Keep original size
    else:
        # Resize using bilinear interpolation (add channel dim temporarily)
        scaled_frame = F.interpolate(frame,
                                    scale_factor=scale,
                                    mode='bilinear',
                                    align_corners=False)
    return scaled_frame


def multi_scale_photometric_loss(scaled_flows, curr_frame, next_frame):
    total_photometric_loss = 0.0
    for idx, scaled_flow in enumerate(scaled_flows):
        # Compute the scale
        scale = 2**(-idx)
        # Get scaled frames
        curr_scaled_frame = get_scaled_frame(curr_frame.float(), scale=scale).cuda()
        next_scaled_frame = get_scaled_frame(next_frame.float(), scale=scale).cuda()
        # Warp current frame
        warped_frame   = warp_frame(curr_scaled_frame.cuda(), scaled_flow.cuda())
        warping_err    = next_scaled_frame.cuda() - warped_frame
        photo_loss     = charbonnier_loss(warping_err)
        total_photometric_loss += photo_loss
    total_photometric_loss = total_photometric_loss / (curr_frame.shape[2]*curr_frame.shape[3])
    return total_photometric_loss
    
    