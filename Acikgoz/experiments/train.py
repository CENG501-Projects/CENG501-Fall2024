import os
from datetime import datetime

import torch
import yaml
from torch import optim
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.tensorboard import SummaryWriter

from datasets.hrsid_loader import load_train
from models.sfs_cnet_model import SFSCNet
from util.metric_util import BBoxLoss, ClassificationLoss, MultiLabelClassificationLoss, match_bboxes, decode_bboxes


def train_model(config_path="../config.yaml", checkpoint_path=None):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Hyperparameters
    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    save_dir = config["training"]["save_dir"]
    input_size = tuple(config["training"]["input_size"])
    num_classes = config["model"]["num_classes"]
    base_channels = config["model"]["base_channels"]

    os.makedirs(save_dir, exist_ok=True)

    # Dataset and DataLoader
    transform = Compose([
        Resize(input_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_loader = load_train(transform=transform, batch_size=batch_size)

    # Model
    model = SFSCNet()
    model = model.to(device := torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Loss functions
    bbox_loss_fn = BBoxLoss()
    cls_loss_fn = MultiLabelClassificationLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))

    # Resume training from checkpoint if provided
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
        epoch_bbox_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_total_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(train_loader, 1):
            images = batch["images"].to(device)
            true_boxes = batch["boxes"]  # List of tensors [Batch_Size, Num_GT, 4]
            true_labels = batch["labels"]  # List of tensors [Batch_Size, Num_GT]

            # Forward pass
            pred_bboxes, pred_classes = model(images)

            decoded_bboxes = decode_bboxes(pred_bboxes, stride=1)

            all_matched_pred_boxes = []
            all_matched_true_boxes = []

            for b in range(decoded_bboxes.size(0)):
                if len(true_boxes[b]) == 0:
                    continue

                true_boxes[b] = true_boxes[b].to(decoded_bboxes.device)

                matched_pred, matched_true = match_bboxes(decoded_bboxes[b], true_boxes[b])
                all_matched_pred_boxes.append(matched_pred)
                all_matched_true_boxes.append(matched_true)


        # Initialize variables for this batch
            all_matched_pred_boxes = []
            all_matched_true_boxes = []
            all_batch_labels = []
            all_batch_pred_cls = []

            # Per-image loss calculation
            batch_size = pred_bboxes.size(0)
            for b in range(batch_size):
                if len(true_boxes[b]) == 0:
                    # No ground truth for this image, skip
                    continue

                # Match predicted boxes to ground truth boxes
                matched_pred, matched_true = match_bboxes(pred_bboxes[b], true_boxes[b])

                all_matched_pred_boxes.append(matched_pred)
                all_matched_true_boxes.append(matched_true)

                gt_labels = true_labels[b].to(pred_classes.device)

                if gt_labels.dim() == 0:
                    gt_labels = gt_labels.unsqueeze(0)

                # Get the first label for this image
                chosen_label = gt_labels[0].long().unsqueeze(0)
                all_batch_labels.append(chosen_label)

                all_batch_pred_cls.append(pred_classes[b][0].unsqueeze(0))  # shape [1, num_classes]

            # Compute bbox loss if any boxes matched
            if len(all_matched_pred_boxes) > 0:
                all_matched_pred = torch.cat(all_matched_pred_boxes, dim=0)  # [Total_Matched, 4]
                all_matched_true = torch.cat(all_matched_true_boxes, dim=0)  # [Total_Matched, 4]
                bbox_loss = bbox_loss_fn(all_matched_pred, all_matched_true)
            else:
                bbox_loss = torch.tensor(0.0, device=pred_bboxes.device)

            # Compute classification loss if any labels are available
            if len(all_batch_labels) > 0:
                cls_targets = torch.cat(all_batch_labels, dim=0)  # [Total_Matched_labels]
                cls_preds = torch.cat(all_batch_pred_cls, dim=0)  # [Total_Matched_labels, num_classes]
                cls_loss = cls_loss_fn(cls_preds, cls_targets)
            else:
                cls_loss = torch.tensor(0.0, device=pred_bboxes.device)

            # Total loss
            total_loss = bbox_loss + cls_loss

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update epoch losses
            epoch_bbox_loss += bbox_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_total_loss += total_loss.item()

            # Log batch details
            print(f"Batch {batch_idx}/{len(train_loader)}")
            print(f"  BBox Loss: {bbox_loss.item():.4f}")
            print(f"  Class Loss: {cls_loss.item():.4f}")
            print(f"  Total Loss: {total_loss.item():.4f}")

            # Log losses to TensorBoard
            writer.add_scalar("Train/Batch BBox Loss", bbox_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar("Train/Batch Class Loss", cls_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar("Train/Batch Total Loss", total_loss.item(), epoch * len(train_loader) + batch_idx)

        # Epoch summary
        avg_bbox_loss = epoch_bbox_loss / len(train_loader)
        avg_cls_loss = epoch_cls_loss / len(train_loader)
        avg_total_loss = epoch_total_loss / len(train_loader)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average BBox Loss: {avg_bbox_loss:.4f}")
        print(f"  Average Class Loss: {avg_cls_loss:.4f}")
        print(f"  Average Total Loss: {avg_total_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f"sfs_cnet_epoch_{epoch + 1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    writer.close()
    print("Training completed.")

def initialize_model_xavier(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded: {checkpoint_path} (Resuming from epoch {start_epoch})")
    return start_epoch


def save_model_checkpoint(model, optimizer, epoch, save_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"sfs_cnet_epoch_{epoch}_{timestamp}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"Model checkpoint saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SFS-CNet for object detection.")
    parser.add_argument("--config", type=str, default="../config.yaml", help="Path to the configuration file.")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint file to resume training.")
    args = parser.parse_args()

    train_model(config_path=args.config, checkpoint_path=args.checkpoint)