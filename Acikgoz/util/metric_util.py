import torch

def compute_iou(pred_boxes, true_boxes):
    if pred_boxes.numel() == 0 or true_boxes.numel() == 0:
        return torch.zeros((pred_boxes.size(0), true_boxes.size(0)), device=pred_boxes.device)

    pred_boxes = pred_boxes.unsqueeze(1)  # Shape [N, 1, 4]
    true_boxes = true_boxes.unsqueeze(0)  # Shape [1, M, 4]

    # Calculate intersection
    inter_x1 = torch.max(pred_boxes[..., 0], true_boxes[..., 0])
    inter_y1 = torch.max(pred_boxes[..., 1], true_boxes[..., 1])
    inter_x2 = torch.min(pred_boxes[..., 2], true_boxes[..., 2])
    inter_y2 = torch.min(pred_boxes[..., 3], true_boxes[..., 3])
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Calculate areas of individual boxes
    pred_areas = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    true_areas = (true_boxes[..., 2] - true_boxes[..., 0]) * (true_boxes[..., 3] - true_boxes[..., 1])

    # Calculate union
    union_area = pred_areas + true_areas - inter_area

    # Compute IoU
    iou = inter_area / union_area.clamp(min=1e-6)
    return iou


def compute_classification_accuracy(pred_classes, true_labels):
    # Argmax to get the predicted class indices
    pred_labels = torch.argmax(pred_classes, dim=1)  # Shape: [num_predictions]

    # Compute accuracy
    correct = (pred_labels == true_labels).sum().item()
    total = true_labels.size(0)

    return correct / total if total > 0 else 0.0

def compute_average_precision(iou_matrix, thresholds=[0.5, 0.75]):
    ap_results = {}
    for threshold in thresholds:
        tp = (iou_matrix > threshold).sum(dim=0)  # Count true positives for each ground truth box
        tp = (tp > 0).sum().item()  # Each GT box matched with at least one prediction is a TP
        ap_results[f'AP{int(threshold * 100)}'] = tp / iou_matrix.size(1) if iou_matrix.size(1) > 0 else 0.0
    return ap_results

def compute_mean_average_precision(iou_matrix, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = [x / 100.0 for x in range(50, 100, 5)]  # 0.5 to 0.95 with step 0.05

    ap_values = []
    for threshold in iou_thresholds:
        tp = (iou_matrix > threshold).sum(dim=0)  # True positives
        tp = (tp > 0).sum().item()  # Each GT box matched with a prediction is a TP
        ap = tp / iou_matrix.size(1) if iou_matrix.size(1) > 0 else 0.0
        ap_values.append(ap)

    return sum(ap_values) / len(ap_values) if ap_values else 0.0

def compute_map(iou_matrices, thresholds=[0.5, 0.75, 0.05]):
    map_results = {}
    for threshold in thresholds:
        aps = [compute_average_precision(iou_matrix, thresholds=threshold) for iou_matrix in iou_matrices]
        map_results[f"AP{int(threshold * 100)}"] = torch.tensor(aps).mean().item()
    map_results["mAP"] = torch.tensor(list(map_results.values())).mean().item()
    return map_results
