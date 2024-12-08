from matplotlib import pyplot as plt


def filter_predictions(pred_boxes, pred_scores, score_threshold):
    filtered_boxes = []
    filtered_scores = []
    for boxes, scores in zip(pred_boxes, pred_scores):
        mask = scores > score_threshold
        filtered_boxes.append(boxes[mask])
        filtered_scores.append(scores[mask])
    return filtered_boxes, filtered_scores

import torch
from torchvision.ops import nms

def apply_nms(boxes, scores, iou_threshold):
    # Ensure inputs are tensors
    if isinstance(boxes, list):
        boxes = torch.stack(boxes)
    if isinstance(scores, list):
        scores = torch.stack(scores)

    indices = nms(boxes, scores, iou_threshold)
    return indices

def process_predictions(pred_boxes, pred_scores, score_threshold, iou_threshold):
    # Filter predictions based on score threshold
    filtered_boxes, filtered_scores = filter_predictions(pred_boxes, pred_scores, score_threshold)

    # If no valid predictions, return empty tensors
    if len(filtered_boxes) == 0 or len(filtered_scores) == 0:
        return torch.empty((0, 4)), torch.empty((0,))

    # Ensure filtered_boxes and filtered_scores are tensors
    filtered_boxes = [box for box in filtered_boxes if box.shape[0] > 0]  # Remove empty tensors
    if len(filtered_boxes) == 0:
        return torch.empty((0, 4)), torch.empty((0,))

    filtered_boxes = torch.cat(filtered_boxes, dim=0)
    filtered_scores = torch.cat([score for score in filtered_scores if score.shape[0] > 0], dim=0)

    # Apply Non-Maximum Suppression
    indices = apply_nms(filtered_boxes, filtered_scores, iou_threshold)
    return filtered_boxes[indices], filtered_scores[indices]

def visualize_predictions(image, true_boxes, predicted_boxes):
    image = image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    plt.imshow(image)

    # Plot true boxes in green
    for box in true_boxes.cpu().numpy():
        plt.gca().add_patch(plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            edgecolor='green', facecolor='none', linewidth=2
        ))

    # Plot predicted boxes in red
    for box in predicted_boxes.cpu().numpy():
        plt.gca().add_patch(plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            edgecolor='red', facecolor='none', linewidth=2
        ))

    plt.title("Green: Ground Truth, Red: Predictions")
    plt.show()