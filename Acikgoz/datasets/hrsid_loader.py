import json
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import Compose, Resize, ToTensor

BASE_DATASET_DIR = os.path.join(os.path.dirname(__file__), "../datasets/hrsid_data")

class HRSIDDataset(Dataset):
    def __init__(self, annotation_file, root_dir=None, transform=None):
        """
        HRSID dataset loader compatible with COCO annotations.
        Args:
            annotation_file (str): Path to the COCO-style annotation JSON file.
            root_dir (str): Root directory containing images.
            transform (callable, optional): Transform to apply to the images.
        """
        self.root_dir = root_dir or BASE_DATASET_DIR
        self.image_dir = os.path.join(self.root_dir, "images")
        self.transform = transform or Compose([ToTensor()])

        # Load annotations
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}
        self.annotations = {img_id: [] for img_id in self.images.keys()}
        for ann in data['annotations']:
            self.annotations[ann['image_id']].append({
                'bbox': ann['bbox'],  # COCO format [x_min, y_min, width, height]
                'category_id': ann['category_id']
            })

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image = Image.open(img_path).convert("RGB")

        # Load annotations
        annotations = self.annotations[img_id]
        boxes, labels = [], []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
            labels.append(ann['category_id'])

        # Handle empty annotations
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)

        # Apply transforms
        if self.transform:
            original_size = image.size[::-1]  # Original size as (H, W)
            image = self.transform(image)
            boxes = self._resize_bboxes(boxes, original_size, (image.shape[1], image.shape[2]))

        return {'image': image, 'boxes': boxes, 'labels': labels}

    @staticmethod
    def _resize_bboxes(bboxes, original_size, target_size):
        orig_h, orig_w = original_size
        target_h, target_w = target_size
        scale_x, scale_y = target_w / orig_w, target_h / orig_h

        bboxes[:, [0, 2]] *= scale_x
        bboxes[:, [1, 3]] *= scale_y
        return bboxes

def collate_fn(batch):
    images = [item['image'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]

    images = torch.stack(images, dim=0)
    return {'images': images, 'boxes': boxes, 'labels': labels}

def load_train(transform=None, batch_size=4, shuffle=True, num_workers=4, pin_memory=True):
    annotation_file = os.path.join(BASE_DATASET_DIR, "annotations", "train2017.json")
    dataset = HRSIDDataset(annotation_file=annotation_file, root_dir=BASE_DATASET_DIR, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)

def load_test(transform=None, batch_size=4, shuffle=False, num_workers=4, pin_memory=True):
    annotation_file = os.path.join(BASE_DATASET_DIR, "annotations", "test2017.json")
    dataset = HRSIDDataset(annotation_file=annotation_file, root_dir=BASE_DATASET_DIR, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)

def load_all(transform=None, batch_size=4, shuffle=True, num_workers=4, pin_memory=True):
    annotation_file = os.path.join(BASE_DATASET_DIR, "annotations", "train_test2017.json")
    dataset = HRSIDDataset(annotation_file=annotation_file, root_dir=BASE_DATASET_DIR, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)

def load_dataset(transform=None):
    """
    Load the complete dataset without splitting into train/val.
    Args:
        transform: Transformations to apply to the dataset.
    Returns:
        HRSIDDataset: The complete dataset.
    """
    annotation_file = os.path.join(BASE_DATASET_DIR, "annotations", "train2017.json")
    dataset = HRSIDDataset(annotation_file=annotation_file, root_dir=BASE_DATASET_DIR, transform=transform)
    return dataset

def get_train_val_loaders(val_ratio=0.1, batch_size=1, num_workers=4, pin_memory=True, seed=42, transform=None):
    """
    Split the dataset into training and validation sets by randomly selecting a percentage for validation.
    Args:
        dataset: The complete dataset.
        val_ratio: Fraction of the dataset to use as validation (e.g., 0.1 for 10%).
        batch_size: Number of samples per batch.
        num_workers: Number of worker threads for data loading.
        pin_memory: Whether to use pinned memory.
        seed: Random seed for reproducibility.
    Returns:
        tuple: (train_loader, val_loader)
    """
    dataset = load_dataset(transform=transform)
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Determine the number of validation samples
    num_val_samples = int(len(indices) * val_ratio)

    # Split the indices into training and validation
    val_indices = indices[:num_val_samples]
    train_indices = indices[num_val_samples:]

    # Create Subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Create DataLoaders
    train_loader_ = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)

    return train_loader_, val_loader

# Example usage
if __name__ == "__main__":
    transform = Compose([Resize((640, 640)), ToTensor()])
    train_loader = load_train(transform=transform, batch_size=4, shuffle=True)

    for batch in train_loader:
        images = batch['images']
        boxes = batch['boxes']
        labels = batch['labels']
        print(f"Images Shape: {images.shape}")
        print(f"Number of Boxes: {len(boxes)}")
        print(f"Number of Labels: {len(labels)}")
        if boxes:
            print(f"Box Sample: {boxes[0]}")
            print(f"Label Sample: {labels[0]}")
        break
