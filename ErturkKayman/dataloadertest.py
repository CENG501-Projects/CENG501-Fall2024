from dataset.kitti_dataset import KITTIDataset
from torch.utils.data import DataLoader
import torch

# Custom collate function to handle variable-sized targets
def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized bounding box tensors.
    Args:
        batch: List of (image, bounding_boxes) tuples
    Returns:
        images: Tensor of shape (batch_size, 3, H, W)
        targets: List of tensors with bounding boxes
    """
    images = [item[0] for item in batch]  # Extract images
    targets = [item[1] for item in batch]  # Extract targets (bounding boxes)
    images = torch.stack(images, dim=0)  # Stack images into a single tensor
    return images, targets

# Initialize dataset and dataloader
dataset = KITTIDataset(root_dir="data/KITTIDataset", split="train")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

# Test the dataloader
for images, targets in dataloader:
    print(f"Batch image shape: {images.shape}")
    print(f"Number of targets in batch: {len(targets)}")
    print(f"Shape of first target tensor: {targets[0].shape}")
    break

