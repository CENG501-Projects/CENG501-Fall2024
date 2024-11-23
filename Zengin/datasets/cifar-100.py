import torchvision.datasets as datasets
import os

def download_cifar100(destination="./datasets/cifar100"):
    os.makedirs(destination, exist_ok=True)
    datasets.CIFAR100(destination, download=True)

if __name__ == "__main__":
    download_cifar100()
