import os
import json
import time
import torch
import yaml
from torchvision.transforms import Compose, ToTensor, Normalize
from datasets.hrsid_loader import load_test
from models.sfs_cnet_model import SFSCNet
from util.metric_util import compute_iou, compute_map
from util.visual_util import process_predictions, visualize_predictions, filter_predictions

def test_model(config_path="../config.yaml", checkpoint_path="../trained_models/sfs_cnet_latest.pth",
               output_metrics="../test_results.json", visualize=False):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    num_classes = config["model"]["num_classes"]
    base_channels = config["model"]["base_channels"]

    # Define dataset and dataloader
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_loader = load_test(transform=transform, batch_size=1, shuffle=False)

    # Load model
    model = SFSCNet(in_channels=3, num_classes=num_classes, base_channels=base_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model checkpoint from {checkpoint_path}")

    model.eval()

    # Metrics
    iou_matrices = []
    ap_results = []
    total_images = 0
    total_time = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader, 1):
            images = batch['images'].to(device)
            true_boxes = batch['boxes']
            true_labels = batch['labels']

            start_time = time.time()
            pred_bboxes, pred_classes = model(images)
            end_time = time.time()

            # Measure inference time
            inference_time = end_time - start_time
            total_time += inference_time

            # Compute confidence scores for each box
            pred_scores, _ = torch.max(pred_classes.softmax(dim=2), dim=2)

            for i in range(images.size(0)):
                tb = true_boxes[i].to(device)
                ts = true_labels[i].to(device)
                pb = pred_bboxes[i]
                ps = pred_scores[i]

                # Process predictions
                filtered_boxes, filtered_scores = process_predictions(pb, ps, score_threshold=0.5, iou_threshold=0.5)

                # Compute IoU matrix
                iou_matrix = compute_iou(filtered_boxes, tb)
                iou_matrices.append(iou_matrix)

                # Visualization
                if visualize:
                    visualize_predictions(images[i], tb, filtered_boxes)

                total_images += 1

            print(f"Processed batch {batch_idx}/{len(test_loader)}")

    map_results = compute_map(iou_matrices, thresholds=[0.5, 0.75] + [0.5 + i * 0.05 for i in range(10)])

    results = {
        "total_images": total_images,
        "mAP": map_results["mAP"],
        "AP50": map_results["AP50"],
        "AP75": map_results["AP75"],
        "inference_time": total_time / total_images if total_images > 0 else 0
    }

    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    with open(output_metrics, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_metrics}")
    print(f"mAP: {results['mAP']:.4f}, AP50: {results['AP50']:.4f}, AP75: {results['AP75']:.4f}")
    print(f"Parameters: {results['parameters'] / 1e6:.2f}M, FLOPs: {results['FLOPs'] / 1e9:.2f}G")
    print(f"Average Inference Time: {results['inference_time']:.4f}s")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SFS-CNet for object detection.")
    parser.add_argument("--config", type=str, default="../config.yaml", help="Path to the configuration file.")
    parser.add_argument("--checkpoint", type=str, default="../models/latest_model.pth",
                        help="Path to the model checkpoint.")
    parser.add_argument("--output", type=str, default="../test_results.json", help="Path to save test metrics.")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of predictions.")
    args = parser.parse_args()

    test_model(config_path=args.config, checkpoint_path=args.checkpoint, output_metrics=args.output, visualize=args.visualize)
