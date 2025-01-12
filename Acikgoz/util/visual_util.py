from matplotlib import pyplot as plt, patches


def visualize_predictions(images, predictions, ground_truths, epoch, save_path=None):
    """
    Visualize predictions and ground truths on the images.

    Args:
        images (Tensor): Batch of images, shape [B, C, H, W].
        predictions (list[dict]): Predicted bounding boxes and labels for each image.
                                  Format: [{"boxes": Tensor, "labels": Tensor}, ...]
        ground_truths (list[dict]): Ground truth boxes and labels for each image.
                                    Format: [{"boxes": Tensor, "labels": Tensor}, ...]
        epoch (int): Current epoch number.
        save_path (str, optional): Path to save the visualization. If None, displays inline.
    """
    # Ensure images are in the range [0, 1] for visualization
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to [B, H, W, C]
    images = (images - images.min()) / (images.max() - images.min())  # Normalize to [0, 1]

    batch_size = len(images)
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))

    if batch_size == 1:
        axes = [axes]  # Ensure axes is iterable

    for i in range(batch_size):
        img = images[i]
        preds = predictions[i]
        gts = ground_truths[i]

        # Show image
        axes[i].imshow(img)
        axes[i].axis("off")

        # Add predictions
        for box, label in zip(preds["boxes"], preds["labels"]):
            x_min, y_min, x_max, y_max = box.tolist()
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor="red", facecolor="none"
            )
            axes[i].add_patch(rect)
            axes[i].text(
                x_min, y_min - 5, f"Pred: {label.item()}", color="red", fontsize=8, bbox=dict(facecolor="white", alpha=0.5)
            )

        # Add ground truth boxes
        for box, label in zip(gts["boxes"], gts["labels"]):
            x_min, y_min, x_max, y_max = box.tolist()
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor="green", facecolor="none"
            )
            axes[i].add_patch(rect)
            axes[i].text(
                x_min, y_max + 5, f"GT: {label.item()}", color="green", fontsize=8, bbox=dict(facecolor="white", alpha=0.5)
            )

    fig.suptitle(f"Epoch {epoch}: Predictions (Red) vs Ground Truths (Green)", fontsize=16)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)