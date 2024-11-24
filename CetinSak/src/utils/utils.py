import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def cifar_10_download(path="./data", transform=None):
    """Download CIFAR10 dataset using torchvision downloaders

    Args:
        path (str or os.path , optional): Overrides torchvision default dataloader location. Defaults to ./data.
        transform (torchvision.transforms.Compose, optional): Override default ToTensor transformation. Can be used for data preprocessing. Defaults to None.

    Returns:
        train_dataset, test_dataset (CIFAR10, CIFAR10) : CIFAR10 datasets for train and test.
    """
    if not transform:
        transform = transforms.Compose(transforms=[transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=path,
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset


def cifar_10_load(dataset, batch_size=64, shuffle=True, num_workers=1):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader


if __name__ == "__main__":
    train_dataset, test_dataset = cifar_10_download()

    train_loader = cifar_10_load(train_dataset)
    test_loader = cifar_10_load(test_dataset, shuffle=False)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    for images, labels in train_loader:
        print(f"Batch size: {images.size()}, Labels: {labels.size()}")
        break
