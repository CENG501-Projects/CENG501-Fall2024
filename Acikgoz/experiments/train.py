import os
import time
import torch
import yaml
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, ToTensor
from datasets.hrsid_loader import get_train_val_loaders
from models.sfs_cnet_model import SFSCNet
from util.metric_util import calculate_losses, calculate_map, scale_boxes
from util.visual_util import visualize_predictions

def initialize_weights_xavier(model):
    """Initialize the model weights using Xavier initialization."""
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

def validate_tensor(tensor, name):
    """Validate if a tensor contains NaN or Inf values."""
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN!")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf!")

def validate_predictions(pred_bboxes, pred_classes):
    """Validate predictions to ensure they are within expected ranges."""
    for i, bbox in enumerate(pred_bboxes):
        validate_tensor(bbox, f"Predicted BBoxes [{i}]")
    for i, cls in enumerate(pred_classes):
        validate_tensor(cls, f"Predicted Classes [{i}]")

def train_model(config_path="../config.yaml", checkpoint_path=None):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Training parameters
    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    momentum = config["training"]["momentum"]
    save_dir = config["training"]["save_dir"]
    log_dir = config["training"]["log_dir"]
    num_classes = config["model"]["num_classes"]
    image_size = tuple(config["training"]["input_size"])

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Define transformations
    transform = Compose([Resize(image_size), ToTensor()])

    train_loader, val_loader = get_train_val_loaders(val_ratio=0.1, batch_size=1,transform=transform)

    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")

    # Initialize model
    model = SFSCNet(use_optimized_fpu=False)
    initialize_weights_xavier(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer and loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    lambda_bbox = config["training"].get("lambda_bbox", 5.0)
    lambda_cls = config["training"].get("lambda_cls", 0.5)

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Resume training from checkpoint
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resumed training from epoch {start_epoch}.")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_bbox_loss, epoch_cls_loss, epoch_total_loss = 0.0, 0.0, 0.0
        start_epoch_time = time.time()

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(train_loader, 1):
            start_batch_time = time.time()

            images = batch["images"].to(device)
            true_boxes = [box.to(device) for box in batch["boxes"]]
            true_labels = [label.to(device) for label in batch["labels"]]

            optimizer.zero_grad()

            try:
                # Forward pass
                pred_bboxes, pred_classes = model(images)
                validate_predictions(pred_bboxes, pred_classes)

                # Compute losses
                losses = calculate_losses(pred_bboxes, pred_classes, true_boxes, true_labels, device=device, image_size=image_size)
                bbox_loss, cls_loss = losses["bbox_loss"], losses["cls_loss"]

                #total_loss = bbox_loss + cls_loss
                total_loss = lambda_bbox * bbox_loss + lambda_cls * cls_loss

                # Backward pass
                total_loss.backward()
                optimizer.step()

                # Accumulate losses
                epoch_bbox_loss += bbox_loss.item()
                epoch_cls_loss += cls_loss.item()
                epoch_total_loss += total_loss.item()

                batch_time = time.time() - start_batch_time
                print(f"Batch {batch_idx}/{len(train_loader)} - Time: {batch_time:.4f}s - BBox Loss: {bbox_loss:.4f}, Class Loss: {cls_loss:.4f}, Total Loss: {total_loss:.4f}")

                # Log to TensorBoard
                writer.add_scalar("Train/Batch BBox Loss", bbox_loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar("Train/Batch Class Loss", cls_loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar("Train/Batch Total Loss", total_loss.item(), epoch * len(train_loader) + batch_idx)

            except ValueError as e:
                print(f"Skipping Batch {batch_idx}/{len(train_loader)}: {e}")

        epoch_time = time.time() - start_epoch_time
        print(f"Epoch {epoch + 1} Summary: Avg BBox Loss: {epoch_bbox_loss / len(train_loader):.4f}, Avg Class Loss: {epoch_cls_loss / len(train_loader):.4f}, Avg Total Loss: {epoch_total_loss / len(train_loader):.4f}, Epoch Time: {epoch_time:.4f}s")

        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f"sfs_cnet_epoch_{epoch + 1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Validation
        # Validation and Visualization
        if epoch % 100 == 0:
            model.eval()
            predictions, ground_truths = [], []
            with torch.no_grad():
                for batch_idx, val_batch in enumerate(val_loader, 1):
                    val_images = val_batch["images"].to(device)
                    val_true_boxes = [box.to(device) for box in val_batch["boxes"]]
                    val_true_labels = [label.to(device) for label in val_batch["labels"]]

                    val_pred_bboxes, val_pred_classes = model(val_images)

                    print(f"\nValidation Batch {batch_idx}/{len(val_loader)}")
                    print(f"Predicted Bounding Boxes (First Sample):\n{val_pred_bboxes[0]}")
                    print(f"Ground Truth Bounding Boxes (First Sample):\n{val_true_boxes[0]}")

                    for i in range(len(val_images)):
                        scaled_pred_bboxes = scale_boxes(val_pred_bboxes[i], image_size=image_size)
                        predictions.append({
                            "boxes": scaled_pred_bboxes.to(device),
                            "labels": val_pred_classes[i].argmax(dim=-1).to(device)
                        })
                        ground_truths.append({
                            "boxes": val_true_boxes[i],
                            "labels": val_true_labels[i]
                        })

                        # Verbose individual sample predictions
                        print(f"Sample {i + 1}:")
                        print(f"  Predicted Boxes: {scaled_pred_bboxes}")
                        print(f"  Predicted Labels: {val_pred_classes[i].argmax(dim=-1)}")
                        print(f"  Ground Truth Boxes: {val_true_boxes[i]}")
                        print(f"  Ground Truth Labels: {val_true_labels[i]}")

                # Visualize the first batch
                if len(predictions) == batch_size:
                    visualize_predictions(
                        val_images.to(device), predictions, ground_truths, epoch=epoch + 1
                    )

            # Calculate and log mAP
            map_metrics = calculate_map(predictions, ground_truths, image_size=image_size, device=device)

            print(f"\nValidation Results - Epoch {epoch + 1}:")
            print(f"AP50: {map_metrics['AP50']:.4f}, AP75: {map_metrics['AP75']:.4f}")
            print(f"Precision: {map_metrics['Precision']:.4f}, Recall: {map_metrics['Recall']:.4f}, "
                  f"F1: {map_metrics['F1']:.4f}, Accuracy: {map_metrics['Accuracy']:.4f}")

            # Log metrics to TensorBoard
            writer.add_scalar("Validation/AP50", map_metrics["AP50"], epoch + 1)
            writer.add_scalar("Validation/AP75", map_metrics["AP75"], epoch + 1)
            writer.add_scalar("Validation/Precision", map_metrics["Precision"], epoch + 1)
            writer.add_scalar("Validation/Recall", map_metrics["Recall"], epoch + 1)
            writer.add_scalar("Validation/F1", map_metrics["F1"], epoch + 1)
            writer.add_scalar("Validation/Accuracy", map_metrics["Accuracy"], epoch + 1)


    writer.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SFS-CNet for object detection.")
    parser.add_argument("--config", type=str, default="../config.yaml", help="Path to the configuration file.")
    parser.add_argument("--checkpoint", type=str, default="../models/latest_model.pth", help="Path to a checkpoint file to resume training.")
    args = parser.parse_args()

    train_model(config_path=args.config, checkpoint_path=args.checkpoint)