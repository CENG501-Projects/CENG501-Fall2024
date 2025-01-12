import os  # For file and directory management
from PIL import Image  # For handling image loading and transformations
import json  # For loading and parsing JSON files (e.g., Cityscapes polygons JSON)
import numpy as np  # For array manipulations and operations
import torch  # For tensor operations and dataset utilities
from torchvision import transforms  # For image transformations
from torchvision.transforms import functional as F  # For custom image tensor operations
from ExtractBoxesandLabels import *
from Augmentations import *
class CityscapesBBoxDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, instance_dir, json_dir, instance_classes, class_to_category,category_map, transform=None):
        
        self.image_dir = image_dir
        self.instance_dir = instance_dir
        self.json_dir = json_dir
        self.instance_classes = instance_classes
        self.class_to_category = class_to_category
        self.category_map=category_map
        self.transform = transform

        # Match image, instance, and JSON files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith("_leftImg8bit.png")])
        self.instance_files = sorted([f for f in os.listdir(instance_dir) if f.endswith("_gtFine_instanceIds.png")])
        self.json_files = sorted([f for f in os.listdir(json_dir) if f.endswith("_gtFine_polygons.json")])

        assert len(self.image_files) == len(self.instance_files) == len(self.json_files), \
            "Mismatch between images, instance files, and JSON files."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        instance_path = os.path.join(self.instance_dir, self.instance_files[idx])
        json_path = os.path.join(self.json_dir, self.json_files[idx])
        image = Image.open(img_path).convert("RGB")
        scale_x, scale_y=ResizeScaleInformations(img=image)
        
        target = extract_bboxes_from_instanceIds(instance_path, self.instance_classes, json_path, self.class_to_category,self.category_map)
        target['boxes']=AdjustBoundingBoxes(target['boxes'], scale_x, scale_y)


        if self.transform:
            image= self.transform(image)
        
        target=validate_and_filter_boxes(target)
        target['boxes'] = torch.tensor(target['boxes'])
        target['labels'] = torch.tensor(target['labels'],dtype=torch.int64 )

        return image, target
