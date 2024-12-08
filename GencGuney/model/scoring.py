import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np

def msp_score(logits):
    """
    Maximum Softmax Probability (MSP) score.
    
    Args:
        logits (torch.Tensor): Logits output from the model.
        
    Returns:
        float: MSP score (higher = more likely ID).
    """
    probabilities = F.softmax(logits, dim=-1)
    return torch.max(probabilities).item()


def energy_score(logits):
    """
    Energy-based score.
    
    Args:
        logits (torch.Tensor): Logits output from the model.
        
    Returns:
        float: Energy score (lower = more likely OOD).
    """
    return torch.logsumexp(logits, dim=-1).item()


def mahalanobis_score(features, mean, precision_matrix):
    """
    Mahalanobis distance score.
    
    Args:
        features (torch.Tensor): Features from the model.
        mean (torch.Tensor): Mean vector of the in-distribution training features.
        precision_matrix (torch.Tensor): Precision matrix (inverse covariance) of the in-distribution training features.
        
    Returns:
        float: Mahalanobis distance score (lower = more likely ID).
    """
    diff = features - mean
    return torch.dot(diff, torch.matmul(precision_matrix, diff)).item()


def knn_score(features, reference_features, k=1):
    """
    K-Nearest Neighbors (KNN) distance score.
    
    Args:
        features (torch.Tensor): Features of the test sample.
        reference_features (torch.Tensor): Features of the reference in-distribution samples.
        k (int): Number of neighbors to consider.
        
    Returns:
        float: Average distance to the k nearest neighbors (higher = more likely OOD).
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(reference_features)
    distances, _ = nbrs.kneighbors(features.unsqueeze(0).cpu().numpy())
    return np.mean(distances)


def compute_score(test_sample, model, scoring_function, feature_extraction=False, **kwargs):
    """
    General function to compute OOD score for a given test sample and model.
    
    Args:
        test_sample (torch.Tensor): Input test sample.
        model (torch.nn.Module): Pre-trained model.
        scoring_function (callable): Scoring function to use (e.g., msp_score, energy_score).
        feature_extraction (bool): Whether to use the feature outputs instead of logits.
        kwargs: Additional arguments for specific scoring functions (e.g., mean for Mahalanobis).
        
    Returns:
        float: OOD score.
    """
    model.eval()
    with torch.no_grad():
        if feature_extraction:
            features = model(test_sample.unsqueeze(0)).squeeze()
        else:
            logits = model(test_sample.unsqueeze(0)).squeeze()
            features = logits

    return scoring_function(features, **kwargs)


# Example usage
if __name__ == "__main__":
    from torchvision.models import resnet18
    import torchvision.transforms as transforms
    from PIL import Image

    # Example setup
    model = resnet18(pretrained=True)
    model.eval()

    # Load an example image
    url = "https://upload.wikimedia.org/wikipedia/commons/4/41/Siberian_Husky_pho.jpg"
    from io import BytesIO
    import requests
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_sample = preprocess(image)

    # Compute scores
    logits = model(test_sample.unsqueeze(0)).squeeze()
    print("MSP Score:", msp_score(logits))
    print("Energy Score:", energy_score(logits))

    # Example Mahalanobis and KNN scores require additional reference data
    # This example demonstrates modularity for more advanced scenarios
