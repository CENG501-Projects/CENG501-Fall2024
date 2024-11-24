import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Path to save dataset
cifar10_path = './datasets/cifar10'

# Download CIFAR-10
cifar10_train = datasets.CIFAR10(root=cifar10_path, train=True, download=True, transform=transforms.ToTensor())
cifar10_test = datasets.CIFAR10(root=cifar10_path, train=False, download=True, transform=transforms.ToTensor())
