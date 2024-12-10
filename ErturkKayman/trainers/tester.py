import sys
sys.path.append('../MonoATT_Project')

import torch
from torch.utils.data import DataLoader
from dataset.kitti_dataset import KITTIDataset
from models.mono3d_detection import MonoATT

# Custom collate function (reuse from trainer.py)
def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

def test():
    # Paths
    data_dir = "data/KITTIDataset"
    checkpoint_path = "checkpoints/monoatt_epoch_2.pth"

    # Hyperparameters
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = KITTIDataset(root_dir=data_dir, split="val")
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn
    )

    # Load model and checkpoint
    model = MonoATT()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Testing loop
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            print(f"Output shape: {outputs.shape}")
            print(f"First batch output: {outputs[0]}")  # Example output visualization
            break

if __name__ == "__main__":
    test()
