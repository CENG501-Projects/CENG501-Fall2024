import torch
import torch.nn as nn

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
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpsampleConcat, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = CBR(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)  # Upsample the feature map
        x = torch.cat([x, skip], dim=1)  # Concatenate along the channel axis
        return self.conv(x)

def decode_grid_bboxes(bboxes, classes, stride=1.0):
    B, C, H, W = bboxes.shape
    if C != 4:
        raise ValueError("Expected 4 channels in bboxes for x_center,y_center,width,height.")

    # Create a coordinate grid
    # Assume each cell corresponds to a position in the original image space.
    # Adjust according to your model's coordinate system.
    y_range = torch.arange(H, device=bboxes.device) * stride
    x_range = torch.arange(W, device=bboxes.device) * stride
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')  # [H,W]

    # Extract predicted parameters
    x_center = bboxes[:, 0, :, :] + x_grid  # [B,H,W]
    y_center = bboxes[:, 1, :, :] + y_grid  # [B,H,W]
    width = bboxes[:, 2, :, :].clamp(min=0) # [B,H,W] ensure non-negative
    height = bboxes[:, 3, :, :].clamp(min=0) # [B,H,W]

    # Convert (x_center, y_center, width, height) -> (x1,y1,x2,y2)
    x1 = x_center - width / 2.0
    y1 = y_center - height / 2.0
    x2 = x_center + width / 2.0
    y2 = y_center + height / 2.0

    # Flatten H*W cells into N
    # [B,H,W] -> [B,H*W]
    x1 = x1.reshape(B, -1)
    y1 = y1.reshape(B, -1)
    x2 = x2.reshape(B, -1)
    y2 = y2.reshape(B, -1)

    # Stack into [B, N, 4]
    decoded_bboxes = torch.stack([x1, y1, x2, y2], dim=2) # [B,N,4]

    # Decode classes: [B,num_classes,H,W] -> [B,num_classes,N]
    B, num_classes, _, _ = classes.shape
    decoded_classes = classes.reshape(B, num_classes, -1)  # [B,num_classes,N]
    decoded_classes = decoded_classes.permute(0, 2, 1)     # [B,N,num_classes]

    return decoded_bboxes, decoded_classes

class DecoupledHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DecoupledHead, self).__init__()
        # Separate branches for bounding box regression and classification
        self.bbox_regressor = nn.Conv2d(in_channels, 4, kernel_size=1)  # [B,4,H,W]
        self.cls_predictor = nn.Conv2d(in_channels, num_classes, kernel_size=1)  # [B,num_classes,H,W]
        self.num_classes = num_classes

    def forward(self, x):
        # x: [B,in_channels,H,W]
        bboxes = self.bbox_regressor(x)  # [B,4,H,W]
        classes = self.cls_predictor(x)  # [B,num_classes,H,W]

        # Decode the grid predictions into a list of boxes [B,N,4] and class scores [B,N,num_classes]
        decoded_bboxes, decoded_classes = decode_grid_bboxes(bboxes, classes)

        return decoded_bboxes, decoded_classes

def match_bboxes(pred_bboxes, true_bboxes):
    if pred_bboxes.dim() == 1 and pred_bboxes.size(0) == 4:
        pred_bboxes = pred_bboxes.unsqueeze(0)
    if true_bboxes.dim() == 1 and true_bboxes.size(0) == 4:
        true_bboxes = true_bboxes.unsqueeze(0)

    if pred_bboxes.numel() == 0:
        pred_bboxes = pred_bboxes.view(-1,4)
    if true_bboxes.numel() == 0:
        true_bboxes = true_bboxes.view(-1,4)

    if pred_bboxes.size(-1) != 4 or true_bboxes.size(-1) != 4:
        print("pred_bboxes shape:", pred_bboxes.shape)
        print("true_bboxes shape:", true_bboxes.shape)
        raise ValueError("Both pred_bboxes and true_bboxes must have shape [N,4].")

    iou = compute_iou(pred_bboxes, true_bboxes) # You must have a compute_iou function defined
    matched_indices = iou.argmax(dim=0)  # best pred for each GT
    matched_pred_bboxes = pred_bboxes[matched_indices]  # shape [num_gt,4]
    matched_true_bboxes = true_bboxes  # shape [num_gt,4]
    return matched_pred_bboxes, matched_true_bboxes

class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        self.bbox_loss_fn = nn.SmoothL1Loss(reduction='mean')
        self.cls_loss_fn = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        pred_bboxes, pred_classes = preds
        true_bboxes_list, true_labels_list = targets

        all_matched_pred_boxes = []
        all_matched_true_boxes = []
        all_batch_labels = []
        all_batch_pred_cls = []

        batch_size = pred_bboxes.size(0)
        for b in range(batch_size):
            if len(true_bboxes_list[b]) == 0:
                # No ground truth for this image, skip
                continue

            # Match predicted boxes to ground truth boxes
            # pred_bboxes[b]: [num_preds,4]
            # true_bboxes_list[b]: [num_gt,4]
            matched_pred, matched_true = match_bboxes(pred_bboxes[b], true_bboxes_list[b])

            all_matched_pred_boxes.append(matched_pred)
            all_matched_true_boxes.append(matched_true)

            gt_labels = true_labels_list[b].to(pred_classes.device)

            if gt_labels.dim() == 0:
                gt_labels = gt_labels.unsqueeze(0)

            chosen_label = gt_labels[0].long().unsqueeze(0)
            all_batch_labels.append(chosen_label)

            all_batch_pred_cls.append(pred_classes[b][0].unsqueeze(0))  # shape [1,num_classes]

        if len(all_matched_pred_boxes) == 0:
            return torch.tensor(0.0, device=pred_bboxes.device)

        all_matched_pred = torch.cat(all_matched_pred_boxes, dim=0)  # [Total_Matched,4]
        all_matched_true = torch.cat(all_matched_true_boxes, dim=0)  # [Total_Matched,4]

        bbox_loss = self.bbox_loss_fn(all_matched_pred, all_matched_true)

        if len(all_batch_labels) == 0:
            cls_loss = torch.tensor(0.0, device=pred_bboxes.device)
        else:
            cls_targets = torch.cat(all_batch_labels, dim=0)          # [Total_Matched_labels]
            cls_preds = torch.cat(all_batch_pred_cls, dim=0)          # [Total_Matched_labels, num_classes]
            cls_loss = self.cls_loss_fn(cls_preds, cls_targets)

        total_loss = bbox_loss + cls_loss
        return total_loss