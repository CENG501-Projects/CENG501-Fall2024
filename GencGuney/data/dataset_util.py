# src/data/data_utils.py

import os
import random
import requests
import tarfile
import zipfile
from tqdm import tqdm
from typing import Tuple

import torch
from torch.utils.data import random_split
import torchvision.transforms as T

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Import your CustomImageDataset class (adjust the path as needed).
from data.datasets_id_ood import CustomImageDataset, CombinedID_OOD_Dataset


# ------------------------------------------------------------------------------
# 1. Transforms
#    Define standard transformations for training and testing sets.
#    Adjust these based on your dataset’s best practices.
# ------------------------------------------------------------------------------

def get_transforms(
    dataset_name: str,
    train: bool = True
) -> T.Compose:
    """
    Returns a PyTorch transform pipeline for a given dataset and mode (train/test).

    Args:
        dataset_name: A string (e.g. 'cifar10', 'cifar100', etc.) used to decide transforms.
        train: Whether to build transforms for training (with augmentation) or testing/validation.

    Returns:
        A torchvision.transforms.Compose object with the appropriate transformations.
    """

    # Example normalization stats for CIFAR-like datasets
    cifar_stats = {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010)
    }

    # Example normalization stats for ImageNet-like datasets (used for Places365, etc.)
    # These might not be exact for your scenario, adjust as needed.
    imagenet_stats = {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225)
    }

    dataset_lower = dataset_name.lower()

    # Decide stats based on dataset
    if dataset_lower in ['cifar10', 'cifar100', 'svhn']:
        stats = cifar_stats
        img_size = 32
    elif dataset_lower in ['places365']:
        stats = imagenet_stats
        img_size = 224
    else:
        # Default to ImageNet stats if unknown
        stats = imagenet_stats
        img_size = 224

    if train:
        # Training transforms (add random augmentations)
        transform = T.Compose([
            T.Resize((img_size, img_size)),           # resize if necessary
            T.RandomHorizontalFlip(p=0.5),            # basic augmentation
            T.RandomCrop(img_size, padding=4),        # typical CIFAR augmentation
            T.ToTensor(),
            T.Normalize(stats['mean'], stats['std'])
        ])
    else:
        # Testing/validation transforms
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(stats['mean'], stats['std'])
        ])

    return transform


# ------------------------------------------------------------------------------
# 2. ID Dataset Creation
#    Functions that create train, val, and test splits for in-distribution data.
#    Adjust for your needs (e.g., custom validation sizes, multiple splits, etc.).
# ------------------------------------------------------------------------------

def create_id_datasets(
    dataset_name: str,
    root_dir: str,
    train_val_split: float = 0.9,
    download: bool = False,
    seed: int = 42
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Creates train, val, and test datasets for an in-distribution (ID) dataset.
    For example, for CIFAR10 or CIFAR100.

    Args:
        dataset_name: Name of the dataset (e.g., 'cifar10', 'cifar100', 'svhn', 'places365', etc.).
        root_dir: Directory where dataset is stored (or should be downloaded).
        train_val_split: Proportion of the training set used for training (the rest is for validation).
        download: Whether to attempt to download the dataset if not present.
        seed: Random seed for splitting.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # 1) Build transform for training and test
    train_transform = get_transforms(dataset_name, train=True)
    test_transform = get_transforms(dataset_name, train=False)

    # 2) Build "train" dataset
    full_train_dataset = CustomImageDataset(
        dataset_name=dataset_name,
        root_dir=root_dir,
        split='train',
        transform=train_transform,
        is_ood=False,
        download=download
    )

    # 3) Optionally split into train & val
    total_train_size = len(full_train_dataset)
    train_size = int(train_val_split * total_train_size)
    val_size = total_train_size - train_size

    # for reproducibility
    random.seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # 4) Build test dataset
    test_dataset = CustomImageDataset(
        dataset_name=dataset_name,
        root_dir=root_dir,
        split='test',
        transform=test_transform,
        is_ood=False,
        download=download
    )

    return train_dataset, val_dataset, test_dataset


# ------------------------------------------------------------------------------
# 3. OOD Dataset Creation
#    A function to create a single OOD dataset (test or otherwise).
#    Typically, you only need one split (test) for OOD, but you can adapt.
# ------------------------------------------------------------------------------

def create_ood_dataset(
    dataset_name: str,
    root_dir: str,
    split: str = 'test',
    download: bool = False,
    ood_label: int = -1
) -> torch.utils.data.Dataset:
    """
    Creates a single OOD dataset with appropriate transforms and flags it as OOD.

    Args:
        dataset_name: Name of the OOD dataset (e.g. 'svhn', 'cifar100' if used as OOD, 'places365', etc.).
        root_dir: Where the OOD dataset is stored or will be downloaded.
        split: The dataset split (e.g. 'train', 'test', 'val'). Usually 'test' for OOD usage.
        download: Download if needed.
        ood_label: The label assigned to OOD samples (default = -1).

    Returns:
        A CustomImageDataset instance with is_ood=True.
    """
    # Typically for OOD test sets, use "test" transforms (no heavy augmentation)
    ood_transform = get_transforms(dataset_name, train=False)

    ood_dataset = CustomImageDataset(
        dataset_name=dataset_name,
        root_dir=root_dir,
        split=split,
        transform=ood_transform,
        is_ood=True,
        ood_label=ood_label,
        download=download
    )

    return ood_dataset


# ------------------------------------------------------------------------------
# 4. Combined Datasets for ID and OOD (if you want a single loader).
# ------------------------------------------------------------------------------

def create_combined_dataset(
    id_dataset: torch.utils.data.Dataset,
    ood_dataset: torch.utils.data.Dataset
) -> CombinedID_OOD_Dataset:
    """
    Combines an ID dataset and an OOD dataset into a single dataset for
    unified loading in a single DataLoader. ID keeps original labels, OOD is -1.

    Args:
        id_dataset: An in-distribution dataset (e.g., CIFAR10 test).
        ood_dataset: An out-of-distribution dataset (e.g., SVHN test).
    
    Returns:
        A CombinedID_OOD_Dataset which concatenates the two.
    """
    combined_dataset = CombinedID_OOD_Dataset(id_dataset, ood_dataset)
    return combined_dataset

# ------------------------------------------------------------------------------
# 5. File Download with Retries and Progress Bar
#    A utility function to download files with a progress bar and retry mechanism.
# ------------------------------------------------------------------------------

def download_file(url, save_path):
    """
    Downloads a file from a given URL, providing a progress bar and retry mechanism.

    Args:
        url: The URL of the file to download.
        save_path: The local file path where the downloaded file will be saved.

    Notes:
        - The function attempts to download the file up to three times in case of network errors.
        - A progress bar is displayed during the download, showing the amount of data downloaded.
        - Raises an exception if the maximum number of retries is exceeded.

    Example:
        download_file("https://example.com/file.tar.gz", "downloads/file.tar.gz")
    """
    max_retries = 3
    attempt = 0
    while attempt < max_retries:
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()  # Raise an error for bad status codes
            total_size = int(response.headers.get('content-length', 0))
            with open(save_path, 'wb') as file, tqdm(
                desc=f"Downloading {os.path.basename(save_path)}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            print(f"Download completed: {save_path}")
            return
        except requests.exceptions.RequestException as e:
            attempt += 1
            print(f"Download failed (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                print("Failed to download after multiple attempts. Please check your URL or connection.")
                raise

# ------------------------------------------------------------------------------
# 6. Archive Extraction
#    A utility function to extract `.tar.gz` or `.zip` archives.
# ------------------------------------------------------------------------------

def extract_archive(archive_path, extract_to):
    """
    Extracts a compressed archive (tar.gz or zip) to a specified directory.

    Args:
        archive_path: The file path of the archive to be extracted.
        extract_to: The directory where the contents of the archive will be extracted.

    Notes:
        - Supports `.tar.gz` and `.zip` formats. Other formats are not handled.
        - Ensures all files are extracted to the specified directory.

    Example:
        extract_archive("downloads/file.tar.gz", "extracted/")
    """
    if archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(path=extract_to)
    print(f"Extracted {os.path.basename(archive_path)} to {extract_to}")


# ------------------------------------------------------------------------------
# 7. Example usage demonstration (not executed if imported as a module).
#    You can remove or keep this as a helpful reference / debug usage.
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: Create CIFAR-10 train, val, test
    train_ds, val_ds, test_ds = create_id_datasets(
        dataset_name='cifar10',
        root_dir='datasets',
        train_val_split=0.8,
        download=False  # set True if you haven't downloaded CIFAR-10 yet
    )
    print(f"CIFAR-10 train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # Example: Create SVHN OOD dataset
    svhn_ood = create_ood_dataset(
        dataset_name='svhn',
        root_dir='datasets',
        split='test',
        download=False
    )
    print(f"SVHN OOD test: {len(svhn_ood)}")

    # Example: Combine CIFAR-10 test + SVHN OOD
    combined_test = create_combined_dataset(test_ds, svhn_ood)
    print(f"Combined test set size: {len(combined_test)}")
