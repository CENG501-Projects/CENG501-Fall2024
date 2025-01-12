import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    Args:
        boxes1 (Tensor): Shape [N, 4] in [xmin, ymin, xmax, ymax].
        boxes2 (Tensor): Shape [M, 4] in [xmin, ymin, xmax, ymax].
    Returns:
        Tensor: IoU matrix of shape [N, M].
    """
    if boxes1.device != boxes2.device:
        raise RuntimeError(
            f"boxes1 is on {boxes1.device} while boxes2 is on {boxes2.device}. Move them to the same device.")

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    union_area = area1[:, None] + area2 - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)

    enclose_x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])

    enclose_area = (enclose_x2 - enclose_x1).clamp(min=0) * (enclose_y2 - enclose_y1).clamp(min=0)

    giou = iou - (enclose_area - union_area) / enclose_area.clamp(min=1e-6)

    return giou

def hungarian_matching(iou_matrix):
    """
    Perform Hungarian algorithm-based matching.
    Args:
        iou_matrix (Tensor): IoU matrix [num_preds, num_gt].
    Returns:
        Tensor: Indices of matched ground truth boxes for each prediction.
    """
    row_ind, col_ind = linear_sum_assignment(-iou_matrix.detach().cpu().numpy())  # Use -IoU to maximize
    matched_idx = torch.full((iou_matrix.size(0),), -1, dtype=torch.long, device=iou_matrix.device)
    matched_idx[row_ind] = torch.tensor(col_ind, device=iou_matrix.device)
    return matched_idx

def giou_loss(pred_boxes, gt_boxes, matched_idx):
    """
    Compute GIoU Loss for bounding box regression.
    Args:
        pred_boxes (Tensor): Predicted bounding boxes [num_preds, 4].
        gt_boxes (Tensor): Ground truth bounding boxes [num_gt, 4].
        matched_idx (Tensor): Indices of matched ground truth for each prediction.
    Returns:
        Tensor: Mean GIoU loss.
    """
    valid_idx = matched_idx >= 0
    pred_boxes = pred_boxes[valid_idx]
    gt_boxes = gt_boxes[matched_idx[valid_idx]]

    if len(pred_boxes) == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred_boxes.device)

    giou = box_iou(pred_boxes, gt_boxes).diag()
    return (1 - giou).mean()

def classification_loss(cls_pred, gt_labels, matched_idx):
    """
    Compute classification loss using cross-entropy.
    """
    valid_idx = matched_idx >= 0
    pred_classes = cls_pred[valid_idx]
    gt_classes = gt_labels[matched_idx[valid_idx]]

    if len(pred_classes) == 0:
        return torch.tensor(0.0, requires_grad=True, device=cls_pred.device)

    return F.cross_entropy(pred_classes, gt_classes)

def scale_boxes(pred_boxes, image_size):
    scaled_boxes = pred_boxes.clone()  # Clone to avoid inplace modification
    scaled_boxes[..., 0] *= image_size[0]  # Scale x-coordinates to image width
    scaled_boxes[..., 1] *= image_size[1]  # Scale y-coordinates to image height
    scaled_boxes[..., 2] *= image_size[0]  # Scale width to image width
    scaled_boxes[..., 3] *= image_size[1]  # Scale height to image height
    return scaled_boxes

def calculate_losses(pred_bboxes, pred_classes, gt_boxes, gt_labels, device=None, image_size=(640, 640)):
    """
    Calculate losses for bounding box regression and classification.
    Args:
        pred_bboxes (Tensor): Predicted bounding boxes in normalized coordinates [B, num_preds, 4].
        pred_classes (Tensor): Predicted class logits [B, num_preds, num_classes].
        gt_boxes (list[Tensor]): Ground truth boxes per image in absolute pixel coordinates.
        gt_labels (list[Tensor]): Ground truth labels per image.
        device (torch.device): Target device for computation.
        image_size (tuple): Input image size (width, height).
    Returns:
        dict: Loss values for GIoU and classification.
    """
    total_bbox_loss = 0.0
    total_cls_loss = 0.0

    for i in range(len(pred_bboxes)):
        if gt_boxes[i].size(0) == 0:
            print(f"No ground truth for batch {i}. Skipping...")
            continue

        gt_boxes[i] = gt_boxes[i].to(device)
        gt_labels[i] = gt_labels[i].to(device)

        if not torch.all(torch.isfinite(pred_bboxes[i])) or not torch.all(torch.isfinite(pred_classes[i])):
            print(f"Invalid predictions in batch {i}. Skipping...")
            continue

        scaled_pred_boxes = scale_boxes(pred_bboxes[i], image_size)
        iou_matrix = box_iou(scaled_pred_boxes, gt_boxes[i])
        if iou_matrix.numel() == 0:
            print(f"No IoU values for batch {i}. Skipping...")
            continue

        matched_idx = hungarian_matching(iou_matrix)
        bbox_loss = giou_loss(scaled_pred_boxes, gt_boxes[i], matched_idx)
        cls_loss = classification_loss(pred_classes[i], gt_labels[i], matched_idx)

        total_bbox_loss += bbox_loss
        total_cls_loss += cls_loss

    return {"bbox_loss": total_bbox_loss, "cls_loss": total_cls_loss}

def calculate_map(predictions, ground_truths, image_size=(640, 640), device=None):
    """
    Calculate AP50, AP75, precision, recall, F1 score, and accuracy.

    Args:
        predictions (list): List of predictions, each containing "boxes" and "labels".
        ground_truths (list): List of ground truths, each containing "boxes" and "labels".
        image_size (tuple): Image size for scaling boxes (width, height).

    Returns:
        dict: Metrics including AP50, AP75, precision, recall, F1 score, and accuracy.
    """
    def compute_iou_metrics(iou_threshold):
        tp, fp, fn = 0, 0, 0

        for preds, gts in zip(predictions, ground_truths):
            pred_boxes = preds["boxes"].to(device)
            pred_labels = preds["labels"].to(device)
            gt_boxes = gts["boxes"].to(device)
            gt_labels = gts["labels"].to(device)

            scaled_pred_boxes = scale_boxes(pred_boxes, image_size)

            if scaled_pred_boxes.size(0) == 0 or gt_boxes.size(0) == 0:
                fp += len(scaled_pred_boxes)
                fn += len(gt_boxes)
                continue

            iou_matrix = box_iou(scaled_pred_boxes, gt_boxes)
            matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)

            for i, pred_box in enumerate(scaled_pred_boxes):
                max_iou, gt_idx = iou_matrix[i].max(0)
                if max_iou >= iou_threshold and not matched_gt[gt_idx] and pred_labels[i] == gt_labels[gt_idx]:
                    tp += 1
                    matched_gt[gt_idx] = True
                else:
                    fp += 1

            fn += (~matched_gt).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

        return precision, recall, f1, accuracy

    def compute_average_precision(iou_threshold):
        tp, fp, fn = 0, 0, 0
        scores, labels = [], []

        for preds, gts in zip(predictions, ground_truths):
            pred_boxes = preds["boxes"].to(device)
            pred_labels = preds["labels"].to(device)
            gt_boxes = gts["boxes"].to(device)
            gt_labels = gts["labels"].to(device)

            scaled_pred_boxes = scale_boxes(pred_boxes, image_size)

            if scaled_pred_boxes.size(0) == 0 or gt_boxes.size(0) == 0:
                fp += len(scaled_pred_boxes)
                fn += len(gt_boxes)
                continue

            iou_matrix = box_iou(scaled_pred_boxes, gt_boxes)
            matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)

            for i, pred_box in enumerate(scaled_pred_boxes):
                max_iou, gt_idx = iou_matrix[i].max(0)
                if max_iou >= iou_threshold and not matched_gt[gt_idx] and pred_labels[i] == gt_labels[gt_idx]:
                    tp += 1
                    matched_gt[gt_idx] = True
                else:
                    fp += 1

        return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    # Compute metrics
    precision, recall, f1, accuracy = compute_iou_metrics(iou_threshold=0.5)
    ap50 = compute_average_precision(iou_threshold=0.5)
    ap75 = compute_average_precision(iou_threshold=0.75)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "AP50": ap50,
        "AP75": ap75,
    }

