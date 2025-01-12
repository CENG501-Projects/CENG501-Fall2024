import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset.noisy_cifar10 import NoisyCIFAR10
from models.resnet_with_dropout import resnet20
from training.train import train_model
from training.evaluate import evaluate_model
import torchvision.datasets as datasets

import matplotlib.pyplot as plt

normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
noises = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
accuracies = []

print("===========================================")
print('Dropout Model')

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
    learning_rate = 0.001

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Training the model...")
    num_epochs = 200
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

    torch.save(model.state_dict(), f"resnet20_dropout_{noise_ratio}.pth")

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1024, shuffle=False,
        num_workers=4, pin_memory=True)

    print("Model loaded for testing.")
    accuracy = evaluate_model(model, test_loader, device)
    accuracies.append(accuracy)

print(accuracies)
# [86.94, 81.61, 79.44, 75.78, 71.97, 65.95, 61.08, 47.85, 33.52, 20.09, 9.66]
x_axis = list(range(0, 101, 10))
plt.figure(figsize=(8, 6))
plt.plot(x_axis, accuracies, marker='o', linestyle='-', linewidth=2)
plt.xlabel("Percentage of Noisy Annotations (%)", fontsize=14)
plt.ylabel("Test Accuracy (%)", fontsize=14)
plt.title("Effect of Noisy Annotations on Model Performance", fontsize=16)
plt.grid(True)
plt.show()
