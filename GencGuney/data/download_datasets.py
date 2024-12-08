import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_and_extract_archive

def download_cifar10(data_dir="data"):
    """
    Download and prepare the CIFAR10 dataset (ID dataset).
    
    Args:
        data_dir (str): Directory to save the dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    datasets.CIFAR10(root=data_dir, train=True, download=True)
    datasets.CIFAR10(root=data_dir, train=False, download=True)
    print("CIFAR10 dataset downloaded and prepared.")


def download_svhn(data_dir="data"):
    """
    Download and prepare the SVHN dataset (OOD dataset).
    
    Args:
        data_dir (str): Directory to save the dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    datasets.SVHN(root=data_dir, split='train', download=True)
    datasets.SVHN(root=data_dir, split='test', download=True)
    print("SVHN dataset downloaded and prepared.")


def download_lsun(data_dir="data"):
    """
    Download and prepare the LSUN dataset (OOD dataset).
    
    Args:
        data_dir (str): Directory to save the dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    # LSUN is a collection of different scenes; we use LSUN (test set only)
    download_and_extract_archive(
        url="http://dl.yf.io/lsun/objects/bedroom_val_lmdb.zip",
        download_root=data_dir,
        extract_root=os.path.join(data_dir, "LSUN"),
        remove_finished=True
    )
    print("LSUN dataset downloaded and prepared.")


def download_isun(data_dir="data"):
    """
    Download and prepare the iSUN dataset (OOD dataset).
    
    Args:
        data_dir (str): Directory to save the dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    url = "https://people.eecs.berkeley.edu/~hendrycks/Imagenet/iSUN.tar.gz"
    download_and_extract_archive(
        url=url,
        download_root=data_dir,
        extract_root=os.path.join(data_dir, "iSUN"),
        remove_finished=True
    )
    print("iSUN dataset downloaded and prepared.")


def download_places365(data_dir="data"):
    """
    Download and prepare the Places365 dataset (OOD dataset).
    
    Args:
        data_dir (str): Directory to save the dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    url = "http://data.csail.mit.edu/places/places365/val_256.tar"
    download_and_extract_archive(
        url=url,
        download_root=data_dir,
        extract_root=os.path.join(data_dir, "Places365"),
        remove_finished=True
    )
    print("Places365 dataset downloaded and prepared.")


def download_texture(data_dir="data"):
    """
    Download and prepare the Texture dataset (OOD dataset).
    
    Args:
        data_dir (str): Directory to save the dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    download_and_extract_archive(
        url=url,
        download_root=data_dir,
        extract_root=os.path.join(data_dir, "Texture"),
        remove_finished=True
    )
    print("Texture dataset downloaded and prepared.")


def download_all(data_dir="data"):
    """
    Download all required datasets for the project.
    
    Args:
        data_dir (str): Directory to save the datasets.
    """
    print(f"Downloading datasets to: {data_dir}")
    download_cifar10(data_dir)
    download_svhn(data_dir)
    download_places365(data_dir)
    download_texture(data_dir)
    print("All datasets downloaded and prepared.")


if __name__ == "__main__":
    # Example usage
    download_all()
