import sys
sys.path.append('../MonoATT_Project')

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.kitti_dataset import KITTIDataset
from models.mono3d_detection import MonoATT

# Custom collate function
def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized bounding box tensors.
    Args:
        batch: List of (image, bounding_boxes) tuples
    Returns:
        images: Tensor of shape (batch_size, 3, H, W)
        targets: List of tensors with bounding boxes
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

# Trainer class
class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion, device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for images, targets in self.dataloader:
            images = images.to(self.device)
            
            # Keep targets as-is (list of tensors)
            targets = [target.to(self.device) for target in targets]

            # Forward pass
            outputs = self.model(images)

            # Dummy loss computation (update this later for 3D detection loss)
            dummy_target = torch.zeros_like(outputs).to(self.device)
            loss = self.criterion(outputs, dummy_target)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)

    def save_checkpoint(self, epoch, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f"monoatt_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)

# Main training script
def main():
    # Paths
    data_dir = "data/KITTIDataset"
    checkpoint_dir = "checkpoints"

    # Hyperparameters
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = KITTIDataset(root_dir=data_dir, split="train")
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=custom_collate_fn  # Use custom collate function
    )

    # Model, optimizer, and loss function
    model = MonoATT()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()  # Replace with a meaningful loss for 3D detection

    # Trainer
    trainer = Trainer(model, dataloader, optimizer, criterion, device)

    # Training loop
    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Save checkpoint
        trainer.save_checkpoint(epoch + 1, checkpoint_dir)

if __name__ == "__main__":
    main()
