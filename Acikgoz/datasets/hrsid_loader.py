import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

BASE_DATASET_DIR = os.path.join(os.path.dirname(__file__), "../datasets/hrsid_data")

class HRSIDDataset(Dataset):
    def __init__(self, annotation_file, root_dir=None, transform=None):
        if root_dir is None:
            root_dir = BASE_DATASET_DIR
        self.image_dir = os.path.join(root_dir, "images")
        self.transform = transform

        # Load annotation JSON
        annotation_file = os.path.abspath(annotation_file)
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}

        self.annotations = {img_id: [] for img_id in self.images.keys()}
        for ann in data['annotations']:
            self.annotations[ann['image_id']].append({
                'bbox': ann['bbox'],  # [x, y, width, height]
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

        # Get annotations for this image
        annotations = self.annotations[img_id]
        boxes = []
        labels = []

        for ann in annotations:
            x, y, w, h = ann['bbox']

            # Convert [x, y, w, h] to [x1, y1, x2, y2]
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])

        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'boxes': boxes,
            'labels': labels
        }

def collate_fn(batch):
    # Batch is a list of dictionaries
    images = [item['image'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Stack images. They must have the same size (H,W) or transforms must ensure a uniform size.
    images = torch.stack(images, dim=0)

    return {
        'images': images,
        'boxes': boxes,   # list of Tensors, each [N,4]
        'labels': labels  # list of Tensors, each [N]
    }

def load_train(transform=None, batch_size=4, shuffle=True):
    annotation_file = os.path.join(BASE_DATASET_DIR, "annotations", "train2017.json")
    dataset = HRSIDDataset(annotation_file=annotation_file, root_dir=BASE_DATASET_DIR, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def load_test(transform=None, batch_size=4, shuffle=False):
    annotation_file = os.path.join(BASE_DATASET_DIR, "annotations", "test2017.json")
    dataset = HRSIDDataset(annotation_file=annotation_file, root_dir=BASE_DATASET_DIR, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def load_all(transform=None, batch_size=4, shuffle=True):
    annotation_file = os.path.join(BASE_DATASET_DIR, "annotations", "train_test2017.json")
    dataset = HRSIDDataset(annotation_file=annotation_file, root_dir=BASE_DATASET_DIR, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def load_inshore(transform=None, batch_size=4, shuffle=True):
    annotation_file = os.path.join(BASE_DATASET_DIR, "inshore_offshore", "inshore.json")
    dataset = HRSIDDataset(annotation_file=annotation_file, root_dir=BASE_DATASET_DIR, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def load_offshore(transform=None, batch_size=4, shuffle=True):
    annotation_file = os.path.join(BASE_DATASET_DIR, "inshore_offshore", "offshore.json")
    dataset = HRSIDDataset(annotation_file=annotation_file, root_dir=BASE_DATASET_DIR, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# Example usage
if __name__ == "__main__":
    from torchvision.transforms import ToTensor

    train_loader = load_train(transform=ToTensor())
    for batch in train_loader:
        images = batch['images']
        boxes = batch['boxes']
        labels = batch['labels']
        print(f"Train Images: {images.shape}, Boxes: {len(boxes)}, Labels: {len(labels)}")
        # Check shapes for boxes and labels of the first sample
        if len(boxes) > 0:
            print("Boxes shape:", boxes[0].shape)
            print("Labels shape:", labels[0].shape)
        break