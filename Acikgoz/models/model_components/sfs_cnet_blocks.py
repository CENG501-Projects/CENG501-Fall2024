import torch
import torch.nn as nn
import torch.nn.functional as F

from util.metric_util import compute_iou


# 1. CBR Block (Convolution, BatchNorm, ReLU)
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# 3. Upsample and Concat operation (for multi-scale feature fusion)
class UpsampleConcat(nn.Module):
    def __init__(self, in_channels, skip_channels):
        super(UpsampleConcat, self).__init__()

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return torch.cat([x, skip], dim=1)

class DecoupledHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DecoupledHead, self).__init__()
        self.bbox_regressor = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.cls_predictor = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    def forward(self, x):
        bboxes = self.bbox_regressor(x)
        classes = self.cls_predictor(x)
        return bboxes, classes


def match_gt_to_preds(gt_boxes, decoded_bboxes, iou_threshold=0.5):
    iou_matrix = compute_iou(decoded_bboxes, gt_boxes)  # Compute IoU between predictions and GT
    matched_indices = iou_matrix.max(dim=1).indices  # Find best matching GT for each prediction
    matched_gt_boxes = gt_boxes[matched_indices]
    return matched_gt_boxes

