import torchvision.datasets as datasets
import os

def download_cifar10(destination="./datasets/cifar10"):
    os.makedirs(destination, exist_ok=True)
    datasets.CIFAR10(destination, download=True)

if __name__ == "__main__":
    download_cifar10()
