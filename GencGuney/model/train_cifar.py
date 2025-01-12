#!/usr/bin/env python3

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Adjust the import paths to match your project's folder structure
from data.dataset_util import create_id_datasets
from model.models import build_model
from tqdm import tqdm


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Trains the model for one epoch on the given train_loader.
    Prints and returns the average loss.
    """
    model.train()
    running_loss = 0.0
    total_steps = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)


    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_steps += 1

    avg_loss = running_loss / total_steps
    return avg_loss


def evaluate(model, data_loader, device):
    """
    Evaluates the model on the given data_loader.
    Returns the accuracy (0-100) and average loss.
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    total_steps = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Predictions & Accuracy
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item()
            total_steps += 1

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / total_steps
    return accuracy, avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 model and save the weights.")
    parser.add_argument("--model_name", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50",
                                 "resnet101", "resnet152", "densenet121"],
                        help="Which model architecture to train.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Initial learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay (L2 penalty).")
    parser.add_argument("--train_val_split", type=float, default=0.9,
                        help="Proportion of training data used for train vs val.")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of classes. (For CIFAR-10, this is 10.)")
    parser.add_argument("--save_dir", type=str, default="weights",
                        help="Directory to save trained model weights.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix random seeds (optional, ensures reproducibility)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # 1. Create train, val, test datasets from CIFAR-10
    train_dataset, val_dataset, test_dataset = create_id_datasets(
        dataset_name='cifar10',
        root_dir='datasets',
        train_val_split=args.train_val_split,
        download=True,  # downloads if not present
        seed=args.seed
    )

    # 2. Build DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # 3. Create the model
    model = build_model(model_name=args.model_name, num_classes=args.num_classes)
    model.to(device)

    # 4. Define loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # (Optional) a learning rate scheduler if you want
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    # 5. Train for the specified number of epochs
    best_val_acc = 0.0
    start_time = time.time()

    print(f"Starting training: {args.model_name} on device={device} ...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"[Epoch {epoch+1:03d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%")

        # Save the model if it has the best validation accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"{args.model_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  >> Best model so far, saved to {save_path}")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.1f} seconds. Best val acc: {best_val_acc:.2f}%")

    # 6. Evaluate on test set
    test_acc, test_loss = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")

    # 7. Save final model weights
    final_save_path = os.path.join(args.save_dir, f"{args.model_name}_final.pth")
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model weights saved to {final_save_path}")


if __name__ == "__main__":
    main()
