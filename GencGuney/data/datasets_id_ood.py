# src/data/datasets.py

import os
from torch.utils.data import Dataset
from torchvision import datasets

# ----------------------------------------------------------------------------
# 1. A helper function to build standard datasets (e.g., CIFAR, SVHN, etc.)
#    If a dataset is not available in torchvision, you can write a custom class
#    or placeholder below.
# ----------------------------------------------------------------------------

def get_dataset(
    name: str,
    root: str,
    split: str = 'train',
    transform=None,
    target_transform=None,
    download: bool = False
):
    """
    Returns a standard torchvision Dataset based on the 'name' argument.
    
    Args:
        name (str): Name of the dataset, e.g. "cifar10", "cifar100", "svhn", "places365", etc.
        root (str): Root directory where the dataset is stored or should be downloaded to.
        split (str): Usually 'train' or 'test'. Some datasets have custom splits (e.g. 'extra' for SVHN).
        transform: Optional transform to apply to the images.
        target_transform: Optional transform to apply to the labels.
        download (bool): Whether to download the dataset if not present.
    """
    name_lower = name.lower()

    if name_lower == 'cifar10':
        is_train = (split == 'train')
        dataset = datasets.CIFAR10(
            root=root,
            train=is_train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
    elif name_lower == 'cifar100':
        is_train = (split == 'train')
        dataset = datasets.CIFAR100(
            root=root,
            train=is_train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
    elif name_lower == 'svhn':
        # For SVHN: split can be 'train', 'test', or 'extra'
        dataset = datasets.SVHN(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
    elif name_lower == 'places365':
        # TorchVision supports 'train-standard', 'train-challenge', 'val', 'test' for Places365
        # We interpret 'train' as 'train-standard' for convenience
        if split == 'train':
            split = 'train-standard'
        else:
            split = 'val'
        try:
            dataset = datasets.Places365(
                root=root,
                split=split,
                small=True,  # small=True will download ~2GB. False is ~50GB.
                download=download,
                transform=transform,
                target_transform=target_transform
            )
        except:
            dataset = datasets.Places365(
                root=root,
                split=split,
                small=True,  # small=True will download ~2GB. False is ~50GB.
                download=not download,
                transform=transform,
                target_transform=target_transform
            )
    elif name_lower == 'lsun':
        dataset = datasets.LSUN(
            root=root,
            classes="test",
            transform=transform,
            target_transform=target_transform,
        )
    elif name_lower == 'isun':
        return ISUNLoader(
            root_dir=os.path.join(root, 'iSUN/iSUN_patches'),
            transform=transform
        )
    elif name_lower == 'texture':
        return DTDLoader(
            root_dir=os.path.join(root, 'dtd'),
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset name '{name}'. Please implement custom logic.")

    return dataset


# ----------------------------------------------------------------------------
# 2. A unified Dataset class that can be used for both ID and OOD data.
#    - If is_ood=True, the target label is replaced with a default OOD label (e.g., -1).
#    - You can also add additional logic if needed.
# ----------------------------------------------------------------------------

class CustomImageDataset(Dataset):
    """
    A generic Dataset wrapper that can handle both ID and OOD data loading.
    Internally uses torchvision's built-in datasets (if available) or 
    custom placeholders for non-standard datasets.
    
    Usage Example:
    --------------
    # ID data (CIFAR-10, training split)
    train_id_dataset = CustomImageDataset(
        dataset_name='cifar10',
        root_dir='datasets',
        split='train',
        is_ood=False,
        transform=transforms.ToTensor()
    )

    # OOD data (SVHN, test split)
    test_ood_dataset = CustomImageDataset(
        dataset_name='svhn',
        root_dir='datasets',
        split='test',
        is_ood=True,
        transform=transforms.ToTensor(),
        ood_label=-1
    )
    """
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        split: str = 'train',
        transform=None,
        target_transform=None,
        is_ood: bool = False,
        ood_label: int = -1,
        download: bool = False
    ):
        """
        Args:
            dataset_name: Name of the dataset (e.g., "cifar10", "cifar100", "svhn", etc.).
            root_dir: Root directory where the dataset is stored or will be downloaded.
            split: Which data split to load (e.g., 'train', 'test', 'val'). 
            transform: Image transform to apply to each sample.
            target_transform: Target transform to apply to labels.
            is_ood: If True, treats the dataset as OOD and overwrites all labels with ood_label.
            ood_label: Label to assign OOD data. Default is -1.
            download: Whether to download the dataset if not already present.
        """
        super().__init__()

        self.dataset_name = dataset_name
        self.split = split
        self.is_ood = is_ood
        self.ood_label = ood_label

        # Build the underlying dataset (torchvision or custom).
        self.dataset = get_dataset(
            name=dataset_name,
            root=root_dir,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        
        # If this dataset is marked as OOD, override the label with self.ood_label
        if self.is_ood:
            label = self.ood_label
        
        return image, label
    
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ISUNLoader(Dataset):
    """
    Custom dataset loader for the iSUN dataset.
    Expects images to be stored in a directory.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directory containing the iSUN images.
            transform: Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # Label is 0 for OOD (default)
    

class DTDLoader(Dataset):
    """
    Custom dataset loader for the Describable Textures Dataset (DTD).
    Expects the dataset to be structured as `dtd/images/<category>/...`.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Root directory of the DTD dataset (should contain 'images' folder).
            transform: Transformations to apply to the images.
        """
        self.root_dir = os.path.join(root_dir, 'images')  # Point to the 'images' directory
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}

        # Read categories from the 'images' directory and assign labels
        for label_idx, category in enumerate(sorted(os.listdir(self.root_dir))):
            category_path = os.path.join(self.root_dir, category)
            if os.path.isdir(category_path):
                self.label_map[category] = label_idx
                for img_file in os.listdir(category_path):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(category_path, img_file))
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ----------------------------------------------------------------------------
# 3. (Optional) Additional helper if you want to combine ID + OOD in a single dataset.
#    This can be handy for certain evaluation settings. The code merges two datasets
#    (one ID, one OOD) into a single dataset with distinct label sets.
# ----------------------------------------------------------------------------

class CombinedID_OOD_Dataset(Dataset):
    """
    A dataset that concatenates an ID dataset and an OOD dataset for certain tasks
    (e.g., building a combined test loader).
    
    Example usage:
    --------------
    cifar10_id = CustomImageDataset("cifar10", root_dir="datasets", split="test", is_ood=False)
    svhn_ood   = CustomImageDataset("svhn",    root_dir="datasets", split="test", is_ood=True, ood_label=-1)
    
    combined_test = CombinedID_OOD_Dataset(cifar10_id, svhn_ood)
    """
    def __init__(self, id_dataset: Dataset, ood_dataset: Dataset):
        super().__init__()
        self.id_dataset = id_dataset
        self.ood_dataset = ood_dataset
        self.id_length = len(id_dataset)
        self.ood_length = len(ood_dataset)
        self.total_length = self.id_length + self.ood_length

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        if index < self.id_length:
            return self.id_dataset[index]
        else:
            return self.ood_dataset[index - self.id_length]