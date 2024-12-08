import os

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, criterion, optimizer, train_dataset, val_dataset, batch_size, device, save_dir = "./trained_model"):
        """
        Trainer class for handling training and validation.

        Args:
            model: PyTorch model to train.
            criterion: Loss function to use.
            optimizer: Optimizer for training.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            batch_size: Batch size for training and validation.
            device: Device to use for training ('cuda' or 'cpu').
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir

        # Data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc="Training", unit="batch", dynamic_ncols=True)
        for idx, (full_images, face_images, context_images, labels) in enumerate(progress_bar):
        # for idx, (full_images, face_images, context_images, labels) in enumerate(self.train_loader):
            assert full_images is not None and face_images is not None and context_images is not None, f"Found None at index {idx}"

            full_images, face_images, context_images, labels = (
                full_images.to(self.device),
                face_images.to(self.device),
                context_images.to(self.device),
                labels.to(self.device),
            )

            # Forward pass
            self.optimizer.zero_grad()
            predictions, factual_fused, counterfactual_fused = self.model(full_images, face_images, context_images)
            #print("PRED: ", predictions)
            # Compute loss
            loss = self.criterion(factual_fused, counterfactual_fused, predictions, labels)
            running_loss += loss.item()
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

        for param_group in self.optimizer.param_groups:
            print(f"Learning rate = {param_group['lr']}")
        return running_loss / len(self.train_loader)

    def validate(self):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validating", unit="batch", dynamic_ncols=True)
            for idx, (full_images, face_images, context_images, labels) in enumerate(progress_bar):
            #for full_images, face_images, context_images, labels in self.val_loader:
                full_images ,face_images, context_images, labels = (
                    full_images.to(self.device),
                    face_images.to(self.device),
                    context_images.to(self.device),
                    labels.to(self.device),
                )

                predictions, factual_fused, counterfactual_fused = self.model(full_images, face_images, context_images)
                loss = self.criterion(factual_fused, counterfactual_fused, predictions, labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(predictions, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total * 100
        return val_loss / len(self.val_loader), accuracy

    def train(self, num_epochs):
        """Run the training and validation loops for the specified number of epochs."""
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Accuracy: {val_accuracy:.2f}%"
            )
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                print(f"Created directory: {self.save_dir}")
            model_path = os.path.join(self.save_dir, f"model_epoch_{epoch}.pth")
            torch.save(self.model.state_dict(), model_path)
            self.plot_losses(save_path=os.path.join(self.save_dir,f"training_validation_losses_{epoch}.jpg"))


    def plot_losses(self, save_path):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Training Loss")
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Losses")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, format="jpg", dpi=300)
        #plt.show()
        #plt.close()

        print(f"Loss plot saved to {os.path.abspath(save_path)}")

        # Close the plot
        #plt.close()
