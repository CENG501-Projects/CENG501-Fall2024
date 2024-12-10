import sys
sys.path.append('../MonoATT_Project')

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.kitti_dataset import KITTIDataset
from models.mono3d_detection import MonoATT
from tqdm import tqdm  # Progress bar library

# Custom collate function
def custom_collate_fn(batch):
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

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        total_loss = 0

        # Initialize progress bar for the current epoch
        with tqdm(total=len(self.dataloader), desc=f"Epoch {epoch}/{total_epochs}", unit="batch") as pbar:
            for images, targets in self.dataloader:
                images = images.to(self.device)
                
                # Keep targets as-is (list of tensors)
                targets = [target.to(self.device) for target in targets]

                # Forward pass
                outputs = self.model(images)
                outputs = outputs.permute(0, 2, 3, 1).reshape(images.size(0), -1, 7)

                # Compute loss for each image in the batch
                batch_loss = 0
                for i, target in enumerate(targets):
                    if target.shape[0] == 0:
                        continue  # Skip if no ground truth

                    pred_params = outputs[i]  # Predicted parameters (N, 7)
                    gt_params = target[:, [6, 7, 8, 4, 5, 3, 10]]  # Extract relevant GT params (x, y, z, h, w, l, theta)

                    # Compute loss for the first object (simplified)
                    batch_loss += self.criterion(pred_params[0], gt_params[0])

                # Backpropagation
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                total_loss += batch_loss.item()
                pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})
                pbar.update(1)

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

    # Training loop with overall progress bar
    with tqdm(total=num_epochs, desc="Overall Training Progress", unit="epoch") as overall_pbar:
        for epoch in range(1, num_epochs + 1):
            avg_loss = trainer.train_epoch(epoch, num_epochs)
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Save checkpoint
            trainer.save_checkpoint(epoch, checkpoint_dir)
            overall_pbar.update(1)

if __name__ == "__main__":
    main()
