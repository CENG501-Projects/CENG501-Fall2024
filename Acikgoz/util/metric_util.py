from torch import nn

class BBoxLoss(nn.Module):
    def __init__(self):
        super(BBoxLoss, self).__init__()
        self.loss_fn = nn.SmoothL1Loss(reduction="mean")
    def forward(self, pred_bboxes, gt_bboxes):
        return self.loss_fn(pred_bboxes, gt_bboxes)


class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
    def forward(self, pred_classes, gt_classes):
        # Reshape predictions and ground truth for CrossEntropyLoss
        pred_classes = pred_classes.view(-1, pred_classes.shape[-1])  # Flatten to [Batch_Size * N, Num_Classes]
        gt_classes = gt_classes.view(-1)  # Flatten to [Batch_Size * N]
        return self.loss_fn(pred_classes, gt_classes)

class MultiLabelClassificationLoss(nn.Module):
    def __init__(self):
        super(MultiLabelClassificationLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pred_classes, gt_classes):
        return self.loss_fn(pred_classes, gt_classes)

def process_predictions(pred_bboxes, pred_scores, score_threshold=0.5, iou_threshold=0.5):
    mask = pred_scores >= score_threshold
    pred_bboxes = pred_bboxes[mask]
    pred_scores = pred_scores[mask]

    if pred_bboxes.size(0) == 0:
        return torch.empty((0, 4)), torch.empty((0,))

    # Apply NMS
    keep_indices = torch.ops.torchvision.nms(pred_bboxes, pred_scores, iou_threshold)
    filtered_boxes = pred_bboxes[keep_indices]
    filtered_scores = pred_scores[keep_indices]

    return filtered_boxes, filtered_scores

import torch

def compute_iou(pred_boxes, true_boxes):
    device = pred_boxes.device  # Ensure both tensors are on the same device
    true_boxes = true_boxes.to(device)
    if pred_boxes.dim() != 2 or pred_boxes.size(-1) != 4:
        raise ValueError(f"Expected pred_boxes to have shape [Num_Preds, 4], got {pred_boxes.shape}")
    if true_boxes.dim() != 2 or true_boxes.size(-1) != 4:
        raise ValueError(f"Expected true_boxes to have shape [Num_True, 4], got {true_boxes.shape}")

    if pred_boxes.size(0) == 0 or true_boxes.size(0) == 0:
        # Return zero IoU matrix if no predictions or ground truth boxes
        return torch.zeros((pred_boxes.size(0), true_boxes.size(0)), device=pred_boxes.device)

    # Calculate intersection
    x1 = torch.max(pred_boxes[:, None, 0], true_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, None, 1], true_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, None, 2], true_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, None, 3], true_boxes[:, 3])

    inter_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Calculate union
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
    union_area = pred_area[:, None] + true_area - inter_area

    # IoU
    iou_matrix = inter_area / union_area
    return iou_matrix



def compute_map(iou_matrices, thresholds=None):
    if thresholds is None:
        thresholds = [0.5, 0.75] + [0.5 + i * 0.05 for i in range(10)]

    aps = []
    f1_scores = []
    precision_recall_per_threshold = []

    for threshold in thresholds:
        tp, fp, fn = 0, 0, 0

        for iou_matrix in iou_matrices:
            if iou_matrix.size(0) == 0 or iou_matrix.size(1) == 0:
                fp += iou_matrix.size(0)
                fn += iou_matrix.size(1)
                continue

            # Match predicted and ground truth boxes based on IoU threshold
            matched = iou_matrix >= threshold
            tp += matched.sum().item()
            fp += matched.size(0) - matched.sum().item()  # Unmatched predictions
            fn += matched.size(1) - matched.sum().item()  # Unmatched ground truths

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1-score: Harmonic mean of precision and recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

        # Average Precision: Precision weighted by recall
        ap = precision * recall
        aps.append(ap)
        precision_recall_per_threshold.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1})

    # Calculate mean Average Precision (mAP)
    mAP = sum(aps) / len(aps)

    return {
        "mAP": mAP,
        "AP50": aps[0],  # IoU threshold = 0.5
        "AP75": aps[1],  # IoU threshold = 0.75
        "F1_scores": f1_scores,
        "precision_recall": precision_recall_per_threshold,
    }

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

def decode_bboxes(pred_bboxes, stride):
    batch_size, _, height, width = pred_bboxes.shape

    # Generate grid cell centers
    yv, xv = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    xv, yv = xv.to(pred_bboxes.device) * stride, yv.to(pred_bboxes.device) * stride

    # Decode bounding box predictions
    x_center = pred_bboxes[:, 0] + xv.unsqueeze(0)  # Add grid center x
    y_center = pred_bboxes[:, 1] + yv.unsqueeze(0)  # Add grid center y
    w = pred_bboxes[:, 2].exp() * stride  # Decode width
    h = pred_bboxes[:, 3].exp() * stride  # Decode height

    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2

    # Reshape to [Batch_Size, Num_Preds, 4]
    decoded_bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view(batch_size, -1, 4)
    return decoded_bboxes

def match_bboxes(pred_bboxes, true_bboxes, iou_threshold=0.5):

    device = true_bboxes.device  # Ensure all tensors are on the same device
    pred_bboxes = pred_bboxes.to(device)

    # Compute IoU
    iou_matrix = compute_iou(pred_bboxes, true_bboxes)

    matched_pred_boxes = []
    matched_true_boxes = []

    # Match based on IoU threshold
    if iou_matrix.numel() > 0:
        max_iou, max_indices = torch.max(iou_matrix, dim=1)
        max_indices = max_indices.to(device)  # Ensure indices are on the same device as true_bboxes

        matched_pred_boxes = pred_bboxes[max_iou > iou_threshold]
        matched_true_boxes = true_bboxes[max_indices[max_iou > iou_threshold]]

    # Return matched boxes or empty tensors
    matched_pred_boxes = matched_pred_boxes if len(matched_pred_boxes) > 0 else torch.empty((0, 4), device=device)
    matched_true_boxes = matched_true_boxes if len(matched_true_boxes) > 0 else torch.empty((0, 4), device=device)

    return matched_pred_boxes, matched_true_boxes
