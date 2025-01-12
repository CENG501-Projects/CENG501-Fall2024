# src/ood/scoring.py

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict


###############################################################################
# 1. MSP and Energy rely on logits, while Mahalanobis and KNN rely on features.
#    We'll define separate scoring functions for each approach (single sample).
###############################################################################

def score_msp(sample_logits: torch.Tensor) -> float:
    """
    Computes the MSP (Maximum Softmax Probability) for a single sample.
    Typically, a higher MSP implies more in-distribution.

    Args:
        sample_logits (torch.Tensor): shape (num_classes,). The logits of the sample.

    Returns:
        float: The MSP score in [0,1].
    """
    probs = F.softmax(sample_logits, dim=0)  # shape: (num_classes,)
    msp_value = probs.max().item()           # scalar in [0,1]
    return msp_value


def score_energy(sample_logits: torch.Tensor, temperature: float = 1.0) -> float:
    """
    Computes the Energy score for a single sample based on logits.
    Typically, a more negative value => more in-distribution.

    E = - logsumexp(z_i / T)

    Args:
        sample_logits (torch.Tensor): shape (num_classes,).
        temperature (float): Temperature scaling factor.

    Returns:
        float: The energy value (unbounded negative/positive).
    """
    scaled_logits = sample_logits / temperature
    # logsumexp is a scalar
    logsumexp_val = torch.logsumexp(scaled_logits, dim=0).item()
    energy_value = -logsumexp_val
    return energy_value


def score_mahalanobis(
    sample_feature: torch.Tensor,
    gaussian_dict: Dict[str, torch.Tensor],
    class_conditional: bool = True
) -> float:
    """
    Computes the Mahalanobis distance-based score for a single sample's feature.

    If class_conditional=True, we compute the distance to each class mean and take
    the minimum. If there is only one mean (e.g., 'num_classes=1'), we compute a single distance.

    Args:
        sample_feature (torch.Tensor): shape (D,). The penultimate-layer embedding.
        gaussian_dict (dict): Contains "means", "precision" (and optionally "cov").
            - means: shape (C, D) or (1, D)
            - precision: shape (D, D) if shared_cov, or (C, D, D) if class-specific
        class_conditional (bool): whether to compute min distance across classes.

    Returns:
        float: The (minimum) Mahalanobis distance. A smaller value => more in-distribution.
    """
    f = sample_feature.cpu().numpy()        # (D,)
    means = gaussian_dict["means"].cpu().numpy()     # (C, D) or (1, D)
    precision = gaussian_dict["precision"].cpu()     # (D, D) or (C, D, D)

    if class_conditional and len(means.shape) == 2 and len(precision.shape) == 2:
        # Single shared covariance, class-conditional means
        distances = []
        prec_np = precision.numpy()  # (D, D)
        for c in range(means.shape[0]):
            diff = f - means[c]
            dist_c = diff @ prec_np @ diff
            distances.append(dist_c)
        min_dist = min(distances)
        return float(min_dist)

    elif class_conditional and len(precision.shape) == 3:
        # Class-specific covariance
        distances = []
        for c in range(means.shape[0]):
            diff = f - means[c]
            prec_c = precision[c].numpy()  # (D, D)
            dist_c = diff @ prec_c @ diff
            distances.append(dist_c)
        min_dist = min(distances)
        return float(min_dist)

    else:
        # Single global mean + shared precision
        diff = f - means[0]
        prec_np = precision.numpy()  # (D, D)
        dist = diff @ prec_np @ diff
        return float(dist)


def score_knn(
    sample_feature: torch.Tensor,
    id_feature_bank: torch.Tensor,
    k: int = 1
) -> float:
    """
    Computes the KNN distance for a single sample. We find the K nearest neighbors
    in the 'id_feature_bank' and return the average distance. If k=1, it's the nearest distance.

    Args:
        sample_feature (torch.Tensor): shape (D,).
        id_feature_bank (torch.Tensor): shape (N, D). The ID dataset's penultimate features.
        k (int): number of neighbors to consider.

    Returns:
        float: The KNN distance. Larger => more OOD.
    """
    sample_np = sample_feature.cpu().numpy()            # shape (D,)
    id_bank_np = id_feature_bank.cpu().numpy()          # shape (N, D)

    # Euclidean distances from sample to all ID points
    diffs = id_bank_np - sample_np                       # shape (N, D)
    dists = np.sum(diffs**2, axis=1)                    # shape (N,)
    dists = np.sqrt(dists)                              # L2 distances

    # Sort and take the average of the k nearest
    nn_indices = np.argsort(dists)[:k]
    knn_dist = dists[nn_indices].mean()
    return float(knn_dist)


###############################################################################
# 2. Fitting Gaussian for Mahalanobis and utility for ID feature distribution.
###############################################################################

def fit_gaussian_to_id_features(
    id_features: torch.Tensor,
    id_labels: Optional[torch.Tensor] = None,
    num_classes: int = 10,
    shared_cov: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Fits a class-conditional (or single) Gaussian to ID features, returning means, covariance, etc.
    
    Args:
        id_features (torch.Tensor): shape (N, D). Embeddings of ID dataset.
        id_labels (torch.Tensor, optional): shape (N,). Class labels for each feature vector.
                                            If None, we treat all features as from a single distribution.
        num_classes (int): Number of classes if doing class-conditional.
        shared_cov (bool): Whether to use a single shared covariance matrix across all classes.

    Returns:
        dict: {
          "means": torch.Tensor of shape (num_classes, D) or (1, D),
          "precision": torch.Tensor of shape (D, D) (if shared_cov=True) or (num_classes, D, D),
          "cov": torch.Tensor (optional),
        }
    """
    id_features_np = id_features.cpu().numpy()
    N, D = id_features_np.shape

    if id_labels is not None:
        id_labels_np = id_labels.cpu().numpy()
    else:
        # If labels not provided, treat as single distribution
        id_labels_np = np.zeros(N, dtype=int)
        num_classes = 1

    means = []
    for c in range(num_classes):
        class_indices = np.where(id_labels_np == c)[0]
        class_feats = id_features_np[class_indices]
        if len(class_feats) > 0:
            class_mean = class_feats.mean(axis=0)
        else:
            class_mean = np.zeros(D)
        means.append(class_mean)
    means = np.stack(means, axis=0)  # (num_classes, D)

    if shared_cov:
        # Single shared covariance
        diffs = []
        for c in range(num_classes):
            class_indices = np.where(id_labels_np == c)[0]
            class_feats = id_features_np[class_indices]
            centered = class_feats - means[c]
            diffs.append(centered)
        all_diffs = np.concatenate(diffs, axis=0)  # (N, D)
        cov = np.cov(all_diffs, rowvar=False)      # (D, D)
        precision = np.linalg.pinv(cov)
        precision = torch.from_numpy(precision).float()

        result = {
            "means": torch.from_numpy(means).float(),
            "precision": precision,
            "cov": torch.from_numpy(cov).float()
        }
    else:
        # Class-specific covariance
        covs, precisions = [], []
        for c in range(num_classes):
            class_indices = np.where(id_labels_np == c)[0]
            class_feats = id_features_np[class_indices]
            centered = class_feats - means[c]
            cov_c = np.cov(centered, rowvar=False)
            prec_c = np.linalg.pinv(cov_c)
            covs.append(cov_c)
            precisions.append(prec_c)

        result = {
            "means": torch.from_numpy(means).float(),
            "precision": torch.from_numpy(np.stack(precisions, axis=0)).float(),
            "cov": torch.from_numpy(np.stack(covs, axis=0)).float()
        }

    return result


###############################################################################
# 3. Compute ID Scores for an entire dataset (needed for p-value or thresholding).
###############################################################################

def compute_id_scores(
    id_logits_or_features: torch.Tensor,
    score_type: str,
    # For Mahalanobis:
    gaussian_dict: Optional[dict] = None,
    class_conditional: bool = True,
    # For KNN:
    id_feature_bank: Optional[torch.Tensor] = None,
    # For Energy:
    temperature: float = 1.0,
    k: int = 1
) -> torch.Tensor:
    """
    Computes the ID scores array for an entire ID dataset (e.g., for thresholding, p-value).
    The data can be either logits or features, depending on the method:

        MSP, Energy -> logits
        Mahalanobis, KNN -> features
    
    Args:
        id_logits_or_features (torch.Tensor): shape (N, C) if MSP/Energy, or (N, D) if Mahalanobis/KNN.
        score_type (str): One of ["msp", "energy", "mahalanobis", "knn"].
        gaussian_dict (dict, optional): For Mahalanobis, the fitted means/precision.
        class_conditional (bool): For Mahalanobis, whether to do class-wise distance or single.
        id_feature_bank (torch.Tensor, optional): For KNN, the same ID features as a bank.
        temperature (float): For Energy-based scoring.
        k (int): For KNN.

    Returns:
        torch.Tensor: shape (N,) of ID scores.
    """
    all_scores = []

    if score_type.lower() in ["msp", "energy"]:
        # Expect shape (N, C)
        if id_logits_or_features.dim() != 2:
            raise ValueError("For MSP/Energy, 'id_logits_or_features' must be (N, num_classes)")
        for i in range(id_logits_or_features.size(0)):
            logits = id_logits_or_features[i]
            if score_type.lower() == "msp":
                val = score_msp(logits)
            else:
                val = score_energy(logits, temperature=temperature)
            all_scores.append(val)

    elif score_type.lower() == "mahalanobis":
        # Expect shape (N, D) + a fitted gaussian_dict
        if gaussian_dict is None:
            raise ValueError("For Mahalanobis, 'gaussian_dict' cannot be None.")
        if id_logits_or_features.dim() != 2:
            raise ValueError("For Mahalanobis, 'id_logits_or_features' must be (N, D).")
        for i in range(id_logits_or_features.size(0)):
            feats = id_logits_or_features[i]
            dist_val = score_mahalanobis(feats, gaussian_dict, class_conditional=class_conditional)
            all_scores.append(dist_val)

    elif score_type.lower() == "knn":
        # Expect shape (N, D) + an ID feature bank
        if id_feature_bank is None:
            raise ValueError("For KNN, 'id_feature_bank' cannot be None.")
        if id_logits_or_features.dim() != 2:
            raise ValueError("For KNN, 'id_logits_or_features' must be (N, D).")
        for i in range(id_logits_or_features.size(0)):
            feats = id_logits_or_features[i]
            knn_val = score_knn(feats, id_feature_bank, k=k)
            all_scores.append(knn_val)

    else:
        raise ValueError(f"Unknown score_type: {score_type}. Expected one of [msp, energy, mahalanobis, knn].")

    return torch.tensor(all_scores, dtype=torch.float)


###############################################################################
# 4. Compute p-value
#    The p-value is a probability measure of how "extreme" a sample score is
#    under the ID distribution. Commonly, we do:
#         p_value = (# of ID scores >= sample_score) / (N + 1)
#    or a variant depending on whether "higher" or "lower" indicates extreme.
###############################################################################

def compute_p_value(sample_score: float, id_scores_array: torch.Tensor, tail: str = "upper") -> float:
    """
    Computes the p-value for a single sample score, given the ID distribution of scores.

    We interpret 'tail' as:
      - "upper": higher scores => more extreme
      - "lower": lower scores => more extreme

    Then, p_value = (# of ID scores that are >= sample_score) / (N + 1)  [if tail="upper"]
    or p_value = (# of ID scores that are <= sample_score) / (N + 1)    [if tail="lower"]

    Args:
        sample_score (float): The OOD detection score for a single sample.
        id_scores_array (torch.Tensor): shape (N,). The ID distribution of scores.
        tail (str): "upper" or "lower".

    Returns:
        float: p-value in [0,1].
    """
    id_scores_np = id_scores_array.cpu().numpy()
    N = id_scores_np.shape[0]

    if tail.lower() == "upper":
        # # of ID scores >= sample_score
        count_extreme = np.sum(id_scores_np >= sample_score)
    elif tail.lower() == "lower":
        # # of ID scores <= sample_score
        count_extreme = np.sum(id_scores_np <= sample_score)
    else:
        raise ValueError(f"Invalid tail='{tail}'. Must be 'upper' or 'lower'.")

    # A standard approach is (count_extreme + 1) / (N + 1) to avoid p=0 or 1 exactly
    p_value = (count_extreme + 1) / (N + 1)
    return float(p_value)


###############################################################################
# 5. Example usage in __main__ (optional demonstration)
###############################################################################

if __name__ == "__main__":
    # Example demonstration
    import torch

    # Let's say we have ID logits for N=5 samples, each with 10 classes.
    id_logits = torch.randn(5, 10)
    # Compute ID MSP scores
    id_msp_scores = compute_id_scores(id_logits, score_type="msp")
    print("ID MSP scores:", id_msp_scores)

    # Suppose we have a single OOD sample's logits
    ood_logits = torch.randn(10)
    # Compute MSP for that sample
    ood_msp_score = score_msp(ood_logits)
    print("OOD MSP Score:", ood_msp_score)

    # Compute a p-value (assume higher MSP => more ID, so an OOD sample is "extreme" if MSP is smaller, => tail="lower")
    # Actually, with MSP, a smaller score is more suspicious => "lower tail"
    p_val_msp = compute_p_value(ood_msp_score, id_msp_scores, tail="lower")
    print("p-value for OOD MSP Score (lower tail):", p_val_msp)

    # Similarly for Mahalanobis (just a short demonstration):
    id_features = torch.randn(5, 32)  # Suppose 32-dim penultimate embeddings
    # Fit Gaussian
    gauss_dict = fit_gaussian_to_id_features(id_features, num_classes=1, shared_cov=True)
    # Compute ID Mahalanobis scores
    id_maha_scores = compute_id_scores(id_features, "mahalanobis", gaussian_dict=gauss_dict, class_conditional=False)
    print("ID Mahalanobis scores:", id_maha_scores)

    # Single OOD sample features
    ood_feat = torch.randn(32)
    ood_maha_score = score_mahalanobis(ood_feat, gauss_dict, class_conditional=False)
    print("OOD Mahalanobis distance:", ood_maha_score)

    # For Mahalanobis, bigger distance => more OOD => "upper" tail
    p_val_maha = compute_p_value(ood_maha_score, id_maha_scores, tail="upper")
    print("p-value for OOD Mahalanobis Score (upper tail):", p_val_maha)
