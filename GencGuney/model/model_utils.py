# src/models/model_utils.py

import os
import torch
import torch.nn as nn

from model.models import build_model
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
###############################################################################
# 1. Loading Pretrained Weights
###############################################################################

def load_pretrained_weights(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """
    Loads pretrained weights from a checkpoint file into a given model.

    Args:
        model (nn.Module): The model to receive the weights.
        checkpoint_path (str): Path to the .pth file containing state_dict.
        device (str): 'cpu' or 'cuda'.

    Returns:
        nn.Module: The model with loaded weights (on the specified device).
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Put in eval mode by default (for inference)
    return model

###############################################################################
# 2. Creating a Model Zoo
###############################################################################

def create_model_zoo(
    model_names,           # list of model name strings, e.g. ["resnet18", "densenet121"]
    weights_dir=None,      # optional directory containing pretrained weights
    device: str = "cpu",
    num_classes: int = 10
) -> dict:
    """
    Creates a dictionary of models (the 'model zoo').
    Optionally loads pretrained weights if `weights_dir` is specified and a matching
    file <model_name>_best.pth or <model_name>_final.pth is found.

    Args:
        model_names (list): List of model names (e.g., ["resnet18", "resnet34", "densenet121"]).
        weights_dir (str): Directory containing the .pth files (if loading pretrained).
        device (str): 'cpu' or 'cuda'.
        num_classes (int): Number of classes for the final layer (CIFAR-10 = 10).

    Returns:
        dict: Dictionary where keys are model names and values are the corresponding nn.Module objects.
    """
    zoo = {}

    for m_name in model_names:
        # 1) Build the model architecture
        model = build_model(model_name=m_name, num_classes=num_classes)
        model.to(device)

        # 2) If weights_dir is specified, look for a checkpoint
        if weights_dir is not None:
            best_ckpt = os.path.join(weights_dir, f"{m_name}_best.pth")
            final_ckpt = os.path.join(weights_dir, f"{m_name}_final.pth")
            if os.path.isfile(best_ckpt):
                print(f"Loading best checkpoint for {m_name} from {best_ckpt}")
                load_pretrained_weights(model, best_ckpt, device)
            elif os.path.isfile(final_ckpt):
                print(f"Loading final checkpoint for {m_name} from {final_ckpt}")
                load_pretrained_weights(model, final_ckpt, device)
            else:
                print(f"No pretrained weights found for {m_name} in {weights_dir}. Using random init.")

        # 3) Store in the zoo dictionary
        zoo[m_name] = model

    return zoo

###############################################################################
# 3. Inference (Generate Logits)
###############################################################################

def model_inference(model: nn.Module, images: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    Runs inference on a batch of images, returning the logits.

    Args:
        model (nn.Module): The model for inference.
        images (torch.Tensor): A batch of images of shape (B, C, H, W).
        device (str): 'cpu' or 'cuda'.

    Returns:
        torch.Tensor: Logits of shape (B, num_classes).
    """
    model.eval()
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)

    return logits

###############################################################################
# 4. Feature Extraction (Generate Penultimate-Layer Features)
###############################################################################

def extract_penultimate_features(model: nn.Module, images: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    Extract penultimate-layer features for ResNet or DenseNet *CIFAR-10 variants*.
    For ResNet: We'll stop after avgpool and flatten (before the final linear layer).
    For DenseNet: We'll stop after the last BN + ReLU + avgpool (before the classifier).

    Args:
        model (nn.Module): The ResNet or DenseNet model (CIFAR variant).
        images (torch.Tensor): A batch of images.
        device (str): 'cpu' or 'cuda'.

    Returns:
        torch.Tensor: Feature vectors of shape (B, feature_dim).
    """

    model.eval()
    images = images.to(device)

    with torch.no_grad():
        # Check if it's a ResNet-like
        if hasattr(model, "layer4") and hasattr(model, "avgpool"):
            x = model.conv1(images)
            x = model.bn1(x)
            x = model.relu(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            return x

        # Check if it's a DenseNet-like
        elif hasattr(model, "features") and hasattr(model, "classifier"):
            features = model.features(images)
            features = nn.functional.relu(features, inplace=True)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
            return features

        else:
            raise ValueError("Model architecture not recognized for penultimate feature extraction.")

###############################################################################
# 5. Extracting ID Logits & Features Over an Entire Dataset
###############################################################################

def extract_id_logits(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Runs the model on the entire dataset (via the dataloader) to collect the logits.
    Used for ID distribution analysis (e.g., for OOD detection calibration).
    
    Args:
        dataloader: A DataLoader over the ID dataset.
        model: The PyTorch model (with final classification layer).
        device: 'cpu' or 'cuda'.

    Returns:
        torch.Tensor: Concatenated logits of shape (N, num_classes), where N = total samples in dataloader.
    """
    model.eval()
    all_logits = []
    tqdm_dataloader = tqdm(dataloader, desc="Extracting logits")
    with torch.no_grad():
        for images, _ in tqdm_dataloader:
            images = images.to(device)
            outputs = model(images)  # shape: (batch_size, num_classes)
            all_logits.append(outputs.cpu())

    # Concatenate into a single tensor
    all_logits = torch.cat(all_logits, dim=0)
    return all_logits


def extract_id_features(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Runs the model (up to penultimate layer) on the entire dataset to collect the features.
    Useful for constructing ID feature distributions used in many OOD detection methods.

    Args:
        dataloader: A DataLoader over the ID dataset.
        model: The PyTorch model (ResNet or DenseNet for CIFAR).
        device: 'cpu' or 'cuda'.

    Returns:
        torch.Tensor: Concatenated feature vectors of shape (N, feature_dim).
    """
    model.eval()
    all_features = []
    tqdm_dataloader = tqdm(dataloader, desc="Extracting features")
    with torch.no_grad():
        for images, _ in tqdm_dataloader:
            images = images.to(device)
            # Extract penultimate-layer features (using function above)
            feats = extract_penultimate_features(model, images, device)
            all_features.append(feats.cpu())

    # Concatenate into a single tensor
    all_features = torch.cat(all_features, dim=0)
    return all_features

###############################################################################
# 6. Optional Example Usage in __main__
###############################################################################

if __name__ == "__main__":
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import numpy as np

    # For demonstration, let's assume we have a small CIFAR-10 set loaded.
    # Typically you'd do something like: from src.data.data_utils import create_id_datasets
    # train_ds, val_ds, test_ds = create_id_datasets(...)
    # Here, let's just use CIFAR-10 test from torchvision for simplicity:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_dataset = torchvision.datasets.CIFAR10(root='datasets', train=False,
                                                transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Build model & load weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("resnet18", num_classes=10)
    # Suppose we have a pretrained checkpoint
    # model = load_pretrained_weights(model, "weights/resnet18_best.pth", device)
    
    # Extract logits
    logits = extract_id_logits(test_loader, model, device=device)
    print("Logits shape:", logits.shape)
    # Convert to numpy if needed
    logits_np = logits.numpy()
    print("Logits as numpy:", logits_np.shape)

    # Extract features
    features = extract_id_features(test_loader, model, device=device)
    print("Features shape:", features.shape)
    features_np = features.numpy()
    print("Features as numpy:", features_np.shape)