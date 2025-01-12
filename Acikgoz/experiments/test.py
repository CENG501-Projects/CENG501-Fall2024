import os
import json
import time
import torch
import yaml
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from datasets.hrsid_loader import load_test
from models.sfs_cnet_model import SFSCNet
from util.metric_util import calculate_map, scale_boxes
from util.visual_util import visualize_predictions

def test_model(config_path="../config.yaml", checkpoint_path="../trained_models/sfs_cnet_latest.pth",
               output_metrics="../test_results.json", visualize=False):
    """
    Test the SFS-CNet model for object detection.

    Args:
        config_path (str): Path to the configuration file.
        checkpoint_path (str): Path to the trained model checkpoint.
        output_metrics (str): Path to save test metrics as JSON.
        visualize (bool): Whether to visualize predictions.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Model parameters
    num_classes = config["model"]["num_classes"]
    input_size = tuple(config["training"]["input_size"])
    image_size = tuple(config["training"]["input_size"])

    # Dataset and DataLoader
    transform = Compose([
        Resize(input_size),
        ToTensor(),
    ])
    test_loader = load_test(transform=transform, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    model = SFSCNet(use_optimized_fpu=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # Initialize metrics
    predictions = []
    ground_truths = []
    total_images = 0
    total_time = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader, 1):
            images = batch["images"].to(device)
            true_boxes = [box.to(device) for box in batch["boxes"]]
            true_labels = [label.to(device) for label in batch["labels"]]

            # Inference
            start_time = time.time()
            pred_bboxes, pred_classes = model(images)
            end_time = time.time()

            inference_time = end_time - start_time
            total_time += inference_time

            for i in range(images.size(0)):
                gt_boxes = true_boxes[i]
                gt_labels = true_labels[i]

                pred_boxes = pred_bboxes[i].view(-1, 4)  # Flatten predictions
                scaled_pred_bboxes = scale_boxes(pred_boxes, image_size=image_size)
                pred_logits = pred_classes[i]
                pred_scores, pred_labels = torch.max(pred_logits.softmax(dim=1), dim=1)

                # Save predictions and ground truths for metrics
                predictions.append({
                    'boxes': scaled_pred_bboxes,
                    'labels': pred_labels,
                    'scores': pred_scores
                })

                ground_truths.append({
                    'boxes': gt_boxes,
                    'labels': gt_labels
                })

                # Visualization (optional)
                if visualize:
                    visualize_predictions(
                        images[i],
                        gt_boxes,
                        scaled_pred_bboxes,
                        pred_scores
                    )

                total_images += 1

            if total_images == 500:
                break
            print(f"Processed batch {batch_idx}/{len(test_loader)} - Avg Inference Time: {inference_time:.4f}s")

    # Compute mAP and other metrics
    map_results_50 = calculate_map(predictions, ground_truths, image_size=image_size, device=device)
    map_results_75 = calculate_map(predictions, ground_truths, image_size=image_size, device=device)

    # Save results
    results = {
        "total_images": total_images,
        "AP50": map_results_50["AP50"],
        "AP75": map_results_75["AP75"],
        "Precision": map_results_50["Precision"],  # Overall precision
        "Recall": map_results_50["Recall"],        # Overall recall
        "F1": map_results_50["F1"],               # Overall F1 score
        "Accuracy": map_results_50["Accuracy"],   # Overall accuracy
        "average_inference_time": total_time / total_images if total_images > 0 else 0,
    }

    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    with open(output_metrics, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {output_metrics}")
    print(f"AP50: {results['AP50']:.4f}, AP75: {results['AP75']:.4f}")
    print(f"Precision: {results['Precision']:.4f}, Recall: {results['Recall']:.4f}, F1: {results['F1']:.4f}, Accuracy: {results['Accuracy']:.4f}")
    print(f"Total Images: {results['total_images']}, Avg Inference Time: {results['average_inference_time']:.4f}s")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SFS-CNet for object detection.")
    parser.add_argument("--config", type=str, default="../config.yaml", help="Path to the configuration file.")
    parser.add_argument("--checkpoint", type=str, default="../models/latest_model.pth",
                        help="Path to the model checkpoint.")
    parser.add_argument("--output", type=str, default="../test_results.json", help="Path to save test metrics.")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of predictions.")
    args = parser.parse_args()

    test_model(config_path=args.config, checkpoint_path=args.checkpoint, output_metrics=args.output,
               visualize=args.visualize)
