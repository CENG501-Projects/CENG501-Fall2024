import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data.preprocess import get_preprocessing_transforms


def get_cifar10_dataloader(data_dir="data", batch_size=64, train=True, num_workers=4):
    """
    Create a DataLoader for the CIFAR10 dataset (ID dataset).

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the DataLoader.
        train (bool): Whether to load the training split.
        num_workers (int): Number of workers for data loading.

    Returns:
        DataLoader: DataLoader for CIFAR10.
    """
    transform = get_preprocessing_transforms(image_size=32)
    dataset = datasets.CIFAR10(
        root=data_dir, train=train, transform=transform, download=False
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)


def get_svhn_dataloader(data_dir="data", batch_size=64, split="test", num_workers=4):
    """
    Create a DataLoader for the SVHN dataset (OOD dataset).

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the DataLoader.
        split (str): Data split ('train' or 'test').
        num_workers (int): Number of workers for data loading.

    Returns:
        DataLoader: DataLoader for SVHN.
    """
    transform = get_preprocessing_transforms(image_size=32)
    dataset = datasets.SVHN(
        root=data_dir, split=split, transform=transform, download=False
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)


def get_custom_dataloader(data_dir, batch_size=64, image_size=224, shuffle=False, num_workers=4):
    """
    Create a DataLoader for custom datasets like LSUN, iSUN, Places365, and Texture.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the DataLoader.
        image_size (int): Target size to resize images.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of workers for data loading.

    Returns:
        DataLoader: DataLoader for custom datasets.
    """
    transform = get_preprocessing_transforms(image_size=image_size)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_dataloaders(data_dir="data", batch_size=64, num_workers=4):
    """
    Utility to load all required datasets for experiments.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for all DataLoaders.
        num_workers (int): Number of workers for data loading.

    Returns:
        dict: Dictionary of DataLoaders for all datasets.
    """
    print("Loading datasets...")
    dataloaders = {
        "cifar10_train": get_cifar10_dataloader(data_dir, batch_size, train=True, num_workers=num_workers),
        "cifar10_test": get_cifar10_dataloader(data_dir, batch_size, train=False, num_workers=num_workers),
        "svhn_test": get_svhn_dataloader(data_dir, batch_size, split="test", num_workers=num_workers),
        "places365_test": get_custom_dataloader(os.path.join(data_dir, "Places365"), batch_size, shuffle=False, num_workers=num_workers),
        "texture_test": get_custom_dataloader(os.path.join(data_dir, "Texture"), batch_size, shuffle=False, num_workers=num_workers),
    }
    print("Datasets loaded successfully.")
    return dataloaders


if __name__ == "__main__":
    # Example usage
    loaders = get_dataloaders(batch_size=32)
    for name, loader in loaders.items():
        print(f"{name}: {len(loader)} batches")
