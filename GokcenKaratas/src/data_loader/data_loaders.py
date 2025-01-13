import pandas as pd
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
import json
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torch

class EmoticDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file).sample(frac=1, random_state=42).reset_index(drop=True)

        self.transform = transform
        self.all_labels = [
            "Peace", "Affection", "Esteem", "Anticipation", "Engagement", "Confidence",
            "Happiness", "Pleasure", "Excitement", "Surprise", "Sympathy", "Doubt/Confusion",
            "Disconnection", "Fatigue", "Embarrassment", "Yearning", "Disapproval", "Aversion",
            "Annoyance", "Anger", "Sensitivity", "Sadness", "Disquietment", "Fear", "Pain", "Suffering"
        ]

        # Extract columns
        self.filenames = self.data['Filename'].tolist()
        self.folders = self.data['Folder'].tolist()
        self.bboxes = self.data['BBox'].apply(lambda x: eval(x) if isinstance(x, str) else x).tolist()

        # Create a mapping from label names to indices
        self.label_mapping = {label: idx for idx, label in enumerate(sorted(self.all_labels))}

        # Convert categorical labels to one-hot encoded tensors
        self.labels = self.data['Categorical_Labels'].apply(self.encode_label).tolist()

    def encode_label(self, labels):
        """Convert list of string labels to one-hot encoded format."""
        # Initialize a one-hot vector
        one_hot = torch.zeros(len(self.label_mapping), dtype=torch.float32)
        # Iterate through all labels in the list and set corresponding indices to 1
        for label in eval(labels) if isinstance(labels, str) else labels:
            if label in self.label_mapping:  # Ensure label exists in mapping
                one_hot[self.label_mapping[label]] = 1.0
        return one_hot

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Get image path
        folder = self.folders[idx]
        img_path = os.path.join(self.root_dir, folder, self.filenames[idx])
        label = self.labels[idx]

        # Load the image
        image = Image.open(img_path).convert("RGB")
        original_image = image.copy()  # Keep the original for the whole image

        # Get bounding box and extract face region
        bbox = self.bboxes[idx]
        face_image = None
        image_without_face = None

        if bbox:
            # Extract face region
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            face_image = image.crop((x1, y1, x2, y2))

            # Zero out the face region in the image
            image_without_face = image.copy()
            draw = ImageDraw.Draw(image_without_face)
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        else:
            print("No bbox found!!!")

        # Apply transformations
        if self.transform:
            whole_image = self.transform(original_image)
            face_only = self.transform(face_image) if face_image else print("no_face_image")
            whole_image_without_face = self.transform(image_without_face) if image_without_face else print("No without_face")
        else:
            whole_image = original_image
            face_only = face_image if face_image else Image.new("RGB", original_image.size)
            whole_image_without_face = image_without_face if image_without_face else Image.new("RGB", original_image.size)

        # Return a consistent format
        return whole_image, face_only, whole_image_without_face, label


class CAER_SDataset(Dataset):
    def __init__(self, root_dir, face_coords_file, transform=None):
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
        temporary_solution_orig = "../data/train/"
        temporary_solution_orig2 = "../data/test/"
        temporary_solution_replace = ""
        img_path = self.image_paths[idx]

        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        original_image = image.copy()  # Keep the original for the whole image

        # Get face coordinates
        face_box = self.face_coords.get(img_path.replace(temporary_solution_orig, temporary_solution_replace).replace(temporary_solution_orig2, temporary_solution_replace))
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
            #print("Returning default")
            return [whole_image, whole_image, whole_image, label]

        return [whole_image, face_only, whole_image_without_face, label]

