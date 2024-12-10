import torch
import torch.nn as nn

class MonoATTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization_loss = nn.SmoothL1Loss()  # For (x, y, z)
        self.dimension_loss = nn.SmoothL1Loss()    # For (h, w, l)
        self.orientation_loss = nn.SmoothL1Loss()  # For yaw (theta)

    def forward(self, pred_params, gt_params):
        """
        Compute loss for 3D bounding box parameters.
        Args:
            pred_params: Predicted parameters (x, y, z, h, w, l, theta) -> shape: (B, 7)
            gt_params: Ground truth parameters -> shape: (B, 7)
        Returns:
            loss: Composite loss
        """
        loc_loss = self.localization_loss(pred_params[:, :3], gt_params[:, :3])  # x, y, z
        dim_loss = self.dimension_loss(pred_params[:, 3:6], gt_params[:, 3:6])  # h, w, l
        ori_loss = self.orientation_loss(pred_params[:, 6], gt_params[:, 6])    # theta

        # Weighted sum of losses
        loss = loc_loss + dim_loss + ori_loss
        return loss
