"""import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from torchvision import transforms
from facenet_pytorch import MTCNN  # For face detection
import torch

class CAER_SDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        Args:
            root_dir (str): Path to the dataset directory (e.g., 'dataset/train' or 'dataset/test').
            transform (callable, optional): Optional transforms to be applied on an image.
            
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # Initialize face detector
        self.mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Load all image paths and labels
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        original_image = image.copy()  # Keep the original for the whole image

        # Detect face
        face_box = self.mtcnn.detect(image)[0]
        face_image = None
        image_without_face = None

        if face_box is not None:
            # Extract face region
            x1, y1, x2, y2 = [int(coord) for coord in face_box[0]]
            face_image = image.crop((x1, y1, x2, y2))

            # Zero out the face region in the image
            image_without_face = image.copy()
            draw = ImageDraw.Draw(image_without_face)
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

        # Apply transformations
        whole_image = self.transform(original_image) if self.transform else original_image
        face_only = self.transform(face_image) if face_image is not None and self.transform else None
        whole_image_without_face = self.transform(image_without_face) if image_without_face is not None and self.transform else None

        # Check for None values and skip the instance if any are found
        if None in [whole_image, face_only, whole_image_without_face]:
            return [whole_image,whole_image,whole_image, label]

        return [whole_image, face_only, whole_image_without_face, label]
"""

import os
import json
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torch

class CAER_SDataset(Dataset):
    def __init__(self, root_dir, face_coords_file, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            face_coords_file (str): Path to the JSON file with saved face coordinates.
            transform (callable, optional): Optional transforms to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # Load face coordinates
        with open(face_coords_file, 'r') as f:
            self.face_coords = json.load(f)

        # Load all image paths and labels
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        temporary_solution_orig = "/data/Workspace/ybkaratas/caers/data/"
        temporary_solution_replace = "./data/"
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        original_image = image.copy()  # Keep the original for the whole image

        # Get face coordinates
        face_box = self.face_coords.get(img_path.replace(temporary_solution_orig, temporary_solution_replace))
        face_image = None
        image_without_face = None

        if face_box:
            # Extract face region
            x1, y1, x2, y2 = [int(coord) for coord in face_box]
            face_image = image.crop((x1, y1, x2, y2))

            # Zero out the face region in the image
            image_without_face = image.copy()
            draw = ImageDraw.Draw(image_without_face)
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

        # Apply transformations
        whole_image = self.transform(original_image) if self.transform else original_image
        face_only = self.transform(face_image) if face_image is not None and self.transform else None
        whole_image_without_face = self.transform(image_without_face) if image_without_face is not None and self.transform else None

        # Check for None values and skip the instance if any are found
        if None in [whole_image, face_only, whole_image_without_face]:
            return [whole_image, whole_image, whole_image, label]

        return [whole_image, face_only, whole_image_without_face, label]
