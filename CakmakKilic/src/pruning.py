import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset.noisy_cifar10 import NoisyCIFAR10
from models.resnet import resnet20
from training.train import train_model
from training.evaluate import evaluate_model
import torchvision.datasets as datasets
import torch.nn.utils.prune as prune

import matplotlib.pyplot as plt

normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
noises = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
accuracies = []
pruning_ratios = [0, 0.01, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5]
print("===========================================")
print('Pruning Model')

for pruning_ratio in pruning_ratios:
    accuracies = []
    for noise_ratio in noises:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = resnet20().to(device)
        model.load_state_dict(torch.load(f"../models/baseline/resnet20{noise_ratio}.pth", weights_only=True))
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name="weight", amount=pruning_ratio, n=1, dim=0)
        
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

        
        model = resnet20().to(device)
        learning_rate = 0.001

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

        print("Training the model...")
        num_epochs = 5
        train_model(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

        torch.save(model.state_dict(), f"resnet20_pruning_{pruning_ratio}_{noise_ratio}.pth")

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

    print("Pruning ratio: ", pruning_ratio)
    print(accuracies)
    # Pruning ratio:  0 (baseline)
    # [85.55, 82.91, 78.57, 78.9, 75.85, 72.51, 67.84, 61.09, 43.66, 25.1, 7.16]
    # Pruning ratio:  0.01
    # [85.5, 82.09, 80.05, 79.12, 77.17, 73.1, 65.56, 58.75, 44.44, 23.72, 8.23]
    # Pruning ratio:  0.02
    # [85.59, 81.87, 79.38, 80.5, 76.59, 73.85, 67.05, 61.03, 43.16, 23.83, 6.88]
    # Pruning ratio:  0.1
    # [84.33, 81.11, 78.33, 77.42, 73.8, 71.21, 63.07, 56.77, 39.88, 21.75, 10.12]
    # Pruning ratio:  0.2
    # [81.84, 78.77, 75.66, 74.03, 71.92, 66.16, 59.2, 54.49, 36.49, 23.12, 9.66]
    # Pruning ratio:  0.3
    # [76.1, 70.24, 64.73, 66.51, 64.46, 56.5, 52.26, 49.53, 34.44, 20.47, 11.16]
    # Pruning ratio:  0.4
    # [66.66, 63.53, 58.27, 58.45, 58.04, 54.73, 47.47, 44.99, 31.88, 20.66, 10.64]
    # Pruning ratio:  0.5
    # [55.46, 58.31, 56.7, 54.89, 54.17, 48.76, 42.4, 40.3, 34.3, 21.77, 9.51]
