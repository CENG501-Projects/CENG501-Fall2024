import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset.noisy_cifar10 import NoisyCIFAR10
from models.resnet import resnet20
from training.train import train_model, train_model_with_validation
from training.evaluate import evaluate_model, evaluate_model_with_loss
from training.utils import create_train_val_split
import torchvision.datasets as datasets

import matplotlib.pyplot as plt

normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
noises = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
accuracies = []
epoch_losses = []
train_epoch_losses = []
print("===========================================")
print('Early Stopping Model')

for noise_ratio in noises:
    full_train_dataset = DataLoader(
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
    
    train_dataset, val_dataset = create_train_val_split(full_train_dataset, val_ratio=0.2)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet20().to(device)
    learning_rate = 0.001

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training the model with {noise_ratio * 100:.0f}% noisy annotations...")
    num_epochs = 200
    epoch_accuracies, epoch_losses, train_epoch_losses = train_model_with_validation(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)
    

    torch.save(model.state_dict(), f"resnet20_early_stopping_{noise_ratio}.pth")

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1024, shuffle=False,
        num_workers=4, pin_memory=True)

    print("Model loaded for testing.")
    accuracy, test_loss = evaluate_model_with_loss(model, test_loader, criterion, device)
    accuracies.append(accuracy)
    print("validation losses", epoch_losses)
    print("train losses", train_epoch_losses)


print(accuracies)
# [77.95, 77.33, 70.89, 66.39, 65.96, 63.78, 58.64, 51.56, 32.19, 6.72, 3.21]
x_axis = list(range(0, 101, 10))
plt.figure(figsize=(8, 6))
plt.plot(x_axis, accuracies, marker='o', linestyle='-', linewidth=2)
plt.xlabel("Percentage of Noisy Annotations (%)", fontsize=14)
plt.ylabel("Test Accuracy (%)", fontsize=14)
plt.title("Effect of Noisy Annotations on Model Performance", fontsize=16)
plt.grid(True)
plt.show()
