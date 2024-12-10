import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
from dataset.kitti_dataset import KITTIDataset
from torch.utils.data import DataLoader
from models.mono3d_detection import MonoATT
import torch

# Custom collate function
def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

def visualize():
    # Paths
    data_dir = "data/KITTIDataset"
    checkpoint_path = "checkpoints/monoatt_epoch_2.pth"

    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = KITTIDataset(root_dir=data_dir, split="val")
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Visualize one image at a time
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # Load model and checkpoint
    model = MonoATT()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Visualization loop
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)

            # Move to CPU for visualization
            image = images[0].cpu()
            output = outputs[0].cpu()

            # Normalize image back for visualization
            image = F.to_pil_image((image * 0.229 + 0.485).clamp(0, 1))  # Unnormalize

            # Display input image
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Input Image")
            plt.axis("off")

            # Display model output as a heatmap
            heatmap = output[0].detach().numpy()  # Take the first output channel
            plt.subplot(1, 2, 2)
            plt.imshow(heatmap, cmap="viridis")
            plt.title("Model Output Heatmap (Channel 0)")
            plt.axis("off")

            plt.show()

            # Visualize only the first batch
            break

if __name__ == "__main__":
    visualize()
