import torch
from torch.utils.data import DataLoader
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

accuracies = []


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
            noisy_label = random.choice([x for x in range(self.num_classes)])
            self.targets[idx] = noisy_label

        print(f"Applied noise to {num_noisy} out of {num_samples} samples.")



# Train the model
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=200):
    model.train()
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


        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluate the model
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
noises = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for noise_ratio in noises:
    train_loader = DataLoader(
        NoisyCIFAR10(
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
        ),
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet20().to(device)
    # learning_rate =  0.1 * 1024 / 128
    learning_rate = 0.001

    criterion = nn.CrossEntropyLoss().cuda()  # Use standard cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    model.train()
    print("Training the model...")
    num_epochs = 200
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

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
    accuracy = evaluate_model(model, test_loader, device)
    accuracies.append(accuracy)

print(accuracies)
x_axis = list(range(0, 101, 10))
plt.figure(figsize=(8, 6))
plt.plot(x_axis, accuracies, marker='o', linestyle='-', linewidth=2)
plt.xlabel("Percentage of Noisy Annotations (%)", fontsize=14)
plt.ylabel("Test Accuracy (%)", fontsize=14)
plt.title("Effect of Noisy Annotations on Model Performance", fontsize=16)
plt.grid(True)
plt.show()