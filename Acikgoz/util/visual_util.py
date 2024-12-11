import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch import Tensor
from torchvision.transforms import functional as F

def visualize_predictions(image, true_boxes, pred_boxes, figsize=(10, 10), title="Predictions vs Ground Truth"):
    if isinstance(image, Tensor):
        image = F.to_pil_image(image)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

    # Draw ground truth boxes in green
    for box in true_boxes:
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

    # Draw predicted boxes in red
    for box in pred_boxes:
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none', linestyle='dashed')
        ax.add_patch(rect)

    plt.show()