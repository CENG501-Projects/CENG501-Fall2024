import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Ana veri kümesi dizini
        transform: Görüntüler için uygulanacak dönüşümler
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img1_dir = os.path.join(root_dir, "img1")
        self.img2_dir = os.path.join(root_dir, "img2")
        self.labels_dir = os.path.join(root_dir, "labels")
        
        self.img1_files = sorted(os.listdir(self.img1_dir))
        self.img2_files = sorted(os.listdir(self.img2_dir))
        self.label_files = sorted(os.listdir(self.labels_dir))

    def __len__(self):
        return len(self.img1_files)

    def __getitem__(self, idx):
        # Birinci resmi yükle
        img1_path = os.path.join(self.img1_dir, self.img1_files[idx])
        img1 = Image.open(img1_path).convert("L")

        # İkinci resmi yükle
        img2_path = os.path.join(self.img2_dir, self.img2_files[idx])
        img2 = Image.open(img2_path).convert("L")

        # Etiket dosyasını yükle
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        label_data = np.load(label_path)
        
        keypoints0 = label_data["keypoints0"]
        keypoints1 = label_data["keypoints1"]
        matches = label_data["matches"]
        match_confidence = label_data["match_confidence"]

        # Görüntülere dönüşüm uygula
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Tensor döndür
        return img1, img2, {
            "keypoints0": keypoints0,
            "keypoints1": keypoints1,
            "matches": matches,
            "match_confidence": match_confidence
        }


def get_dataloaders(root_dir, batch_size=8, num_workers=4, train_split=0.8, val_split=0.1, seed=42):
    """
    Veri kümesini train, validation ve test olarak böler ve DataLoader'ları döndürür
    """
    transform = transforms.Compose([
        transforms.Resize((832, 832)),
        transforms.ToTensor()
    ])

    # Tüm veri kümesi
    full_dataset = CustomDataset(root_dir, transform=transform)
    dataset_size = len(full_dataset)

    # Split boyutlarını hesapla
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Rastgeleliği kontrol etmek için seed belirle
    generator = torch.Generator().manual_seed(seed)

    # Veri kümesini böl
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # DataLoader oluştur
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

# Örnek kullanım:
# root_dir = "path/to/dataset"
# train_loader, val_loader, test_loader = get_dataloaders(root_dir)
