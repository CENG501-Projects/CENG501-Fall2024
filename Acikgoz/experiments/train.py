import os
from datetime import datetime

import torch
import yaml
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from datasets.hrsid_loader import load_train
from models.model_components.sfs_cnet_blocks import DetectionLoss
from models.sfs_cnet_model import SFSCNet


def train_model(config_path="../config.yaml", checkpoint_path=None):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Read hyperparameters
    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    momentum = config["training"]["momentum"]
    save_dir = config["training"]["save_dir"]
    input_size = tuple(config["training"]["input_size"])
    num_classes = config["model"]["num_classes"]
    base_channels = config["model"]["base_channels"]

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define dataset and dataloader
    transform = Compose([
        Resize(input_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader = load_train(transform=transform, batch_size=batch_size)

    # Initialize model
    model = SFSCNet(in_channels=3, num_classes=num_classes, base_channels=base_channels)
    initialize_model_xavier(model)

    # Initialize loss function and optimizer
    loss_fn = DetectionLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on device: {device}")

    # Resume from checkpoint if provided
    start_epoch = 0
    if checkpoint_path:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
        print(f"Resumed training from checkpoint: {checkpoint_path} (Epoch {start_epoch})")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Starting...")

        for batch_idx, batch in enumerate(train_loader, 1):
            images = batch['images'].to(device)
            boxes = [b.to(device) for b in batch['boxes']]
            labels = [l.to(device) for l in batch['labels']]

            # Forward pass
            pred_bboxes, pred_classes = model(images)

            # Combine boxes and labels into tensors for loss computation
            all_boxes = torch.cat(boxes, dim=0)
            all_labels = torch.cat(labels, dim=0)

            # Compute loss
            loss = loss_fn((pred_bboxes, pred_classes), (all_boxes, all_labels))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Log batch details
            print(f"  Batch [{batch_idx}/{len(train_loader)}]: Loss = {loss.item():.4f}, "
                  f"Boxes = {all_boxes.size(0)}, Labels = {all_labels.size(0)}")

        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Completed: Avg Loss = {avg_loss:.4f}")

        # Save model checkpoint every epoch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"sfs_cnet_epoch_{epoch+1}_{timestamp}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print(f"Model saved to {save_path}")


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SFS-CNet for object detection.")
    parser.add_argument("--config", type=str, default="../config.yaml", help="Path to the configuration file.")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint file to resume training.")
    args = parser.parse_args()

    train_model(config_path=args.config, checkpoint_path=args.checkpoint)
