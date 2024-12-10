from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np

class KITTIDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Root directory of KITTI dataset.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied.
        """
        # Use a fixed transformation for resizing and preprocessing
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # Ensure consistent image size
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])

        self.root_dir = root_dir
        self.split = split
        self.images = []
        self.labels = []

        # Load file names
        with open(os.path.join(root_dir, "ImageSets", f"{split}.txt"), "r") as f:
            file_names = f.read().splitlines()

        # Collect image paths and labels
        for name in file_names:
            image_path = os.path.join(root_dir, "training", "image_2", f"{name}.png")
            label_path = os.path.join(root_dir, "training", "label_2", f"{name}.txt")
            if os.path.exists(image_path) and os.path.exists(label_path):
                self.images.append(image_path)
                self.labels.append(label_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and apply transformation
        image = Image.open(self.images[idx]).convert("RGB")
        image = self.transform(image)

        # Load label
        label_path = self.labels[idx]
        with open(label_path, "r") as f:
            labels = f.read().strip().split("\n")

        # Parse labels (3D bounding boxes)
        bboxes = []
        for label in labels:
            parts = label.split(" ")
            if len(parts) >= 15:  # Ensure sufficient data
                bbox = [float(x) for x in parts[4:15]]  # Extract bbox details
                bboxes.append(bbox)

        bboxes = np.array(bboxes, dtype=np.float32)
        return image, torch.tensor(bboxes)
