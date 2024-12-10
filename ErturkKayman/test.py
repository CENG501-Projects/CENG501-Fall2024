from dataset.kitti_dataset import KITTIDataset
from torchvision import transforms

# KITTI root directory
kitti_root = "data/KITTIDataset"

# Create dataset instance
dataset = KITTIDataset(root_dir=kitti_root, split="train", transform=transforms.ToTensor())

# Test loader
for i in range(3):  # Test with first three images
    image, bboxes = dataset[i]
    print(f"Image shape: {image.shape}, Bounding boxes: {bboxes}")

