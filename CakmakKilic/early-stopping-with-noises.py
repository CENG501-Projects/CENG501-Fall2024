import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch import nn, optim

import os
import pickle
import numpy as np
import torchvision.datasets as datasets
import random
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from resnet import resnet20
import copy


class NoisyCIFAR10(CIFAR10):
    def __init__(self, *args, noise_ratio=0.1, num_classes=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_ratio = noise_ratio
        self.num_classes = num_classes
        self.apply_label_noise()

    def apply_label_noise(self):
        num_samples = len(self.targets)
        num_noisy = int(num_samples * self.noise_ratio)

        noisy_indices = random.sample(range(num_samples), num_noisy)

        for idx in noisy_indices:
            original_label = self.targets[idx]
            noisy_label = random.choice([x for x in range(self.num_classes) if x != original_label])
            self.targets[idx] = noisy_label

        print(f"Applied noise to {num_noisy} out of {num_samples} samples.")


def train_model_with_validation(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=200):
    model.train()
    epoch_accuracies = []
    epoch_losses = []
    train_epoch_losses = []

    best_loss = float('inf')
    best_model_weights = None
    patience = 10

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate validation accuracy after each epoch
        val_accuracy, val_loss = evaluate_model_with_loss(model, val_loader, criterion, device)

        epoch_accuracies.append(val_accuracy)
        epoch_losses.append(val_loss)
        train_epoch_losses.append(running_loss / len(train_loader))
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here
            patience = 10  # Reset patience counter
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break


    model.load_state_dict(best_model_weights)
    return epoch_accuracies, epoch_losses, train_epoch_losses


def evaluate_model_with_loss(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(val_loader)
    return accuracy, avg_loss


def create_train_val_split(dataset, val_ratio=0.1):
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    return random_split(dataset, [train_size, val_size])


accuracies = []
epoch_losses = []
train_epoch_losses = []
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
noises = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for noise_ratio in noises:
    full_train_dataset = NoisyCIFAR10(
        root='./data',
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True,
        noise_ratio=noise_ratio
    )

    train_dataset, val_dataset = create_train_val_split(full_train_dataset, val_ratio=0.2)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet20().to(device)
    # learning_rate =  0.1 * 1024 / 128
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss().cuda()  # Use standard cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    print(f"Training the model with {noise_ratio * 100:.0f}% noisy annotations...")
    num_epochs = 200
    epoch_accuracies, epoch_losses, train_epoch_losses = train_model_with_validation(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)


    torch.save(model.state_dict(), f"resnet20{noise_ratio}.pth")

    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=1024, shuffle=False,
            num_workers=4, pin_memory=True)

    model.eval()  # Set model to evaluation mode
    print("Model loaded for testing.")
    accuracy, test_loss = evaluate_model_with_loss(model, test_loader, criterion, device)
    accuracies.append(accuracy)

    print("validation losses", epoch_losses)
    print("train losses", train_epoch_losses)
    # x_axis = list(range(1, len(epoch_losses) + 1))
    # plt.figure(figsize=(8, 6))
    # plt.plot(x_axis, epoch_losses, marker='o', linestyle='-', linewidth=2, label="Validation Loss")
    # plt.plot(x_axis, train_epoch_losses, marker='x', linestyle='-', linewidth=2, label="Training Loss")
    # plt.xlabel("Number of Iterations (Epochs)", fontsize=14)
    # plt.ylabel("Loss", fontsize=14)
    # plt.title("Training and Validation Loss", fontsize=16)
    # plt.grid(True)
    # plt.show()





print(accuracies)
x_axis = list(range(0, 101, 10))
plt.figure(figsize=(8, 6))
plt.plot(x_axis, accuracies, marker='o', linestyle='-', linewidth=2)
plt.xlabel("Percentage of Noisy Annotations (%)", fontsize=14)
plt.ylabel("Test Accuracy (%)", fontsize=14)
plt.title("Effect of Noisy Annotations on Model Performance", fontsize=16)
plt.grid(True)
plt.show()






