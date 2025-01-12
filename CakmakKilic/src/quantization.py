import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quant
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import random
import numpy as np
import os
from dataset.noisy_cifar10 import NoisyCIFAR10
from models.resnet_qat import create_quantizable_resnet20
from training.train import train_model
from training.evaluate import evaluate_model
from training.qat import fine_tune_and_evaluate_qat


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Info] Using device: {device}")

normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2470, 0.2435, 0.2616]
)

noises = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
qat_accuracies = []
non_qat_accuracies = []
for noise_ratio in noises:
    print("=============================================")
    print(f"[Info] Noise Ratio = {noise_ratio}")

    # 1) Build data loaders
    train_dataset = NoisyCIFAR10(
        root="./data",
        train=True,
        download=True,
        noise_ratio=noise_ratio,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])
    )
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    )

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    non_quantized_model = create_quantizable_resnet20(num_classes=10)
    non_quantized_model = non_quantized_model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(non_quantized_model.parameters(), lr=1e-3)
    epochs = 60

    print("Train model from scratch")

    non_quantized_model.train()
    train_model(non_quantized_model, train_loader, criterion, optimizer, device, epochs)

    non_qat_accuracy = evaluate_model(non_quantized_model, test_loader, device)
    non_qat_accuracies.append(non_qat_accuracy)
    
    torch.save(non_quantized_model.state_dict(), f'non_quantized_model_{noise_ratio}.pth')

    _, acc = fine_tune_and_evaluate_qat(
        non_quantized_model,
        train_loader,
        test_loader,
        device=device,
        epochs=5,    
        lr=1e-4,
        noise_ratio=noise_ratio
    )
    qat_accuracies.append(acc)

print("=================================================")
print("[Summary] QAT Accuracies:", qat_accuracies)
