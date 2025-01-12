#!/usr/bin/env python3
"""
zode_pipeline.py

Implements the ZODE detection procedure using the Benjamini-Hochberg method
for p-values across multiple models (the model zoo). Integrates TPR, FPR, and
AUC computation for better interpretation of results.
"""

import os
import sys
import argparse
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------------------------------
# Adjust the following imports to match your repository structure:
# ------------------------------------------------------------------------------
from data.dataset_util import create_id_datasets, create_ood_dataset
from model.model_utils import create_model_zoo, extract_id_logits, extract_id_features
from zode.scoring import (
    compute_id_scores,
    score_msp,
    score_energy,
    score_mahalanobis,
    score_knn,
    compute_p_value,
    fit_gaussian_to_id_features
)


# ------------------------------------------------------------------------------
# 0. Utility function: Benjamini-Hochberg Procedure
# ------------------------------------------------------------------------------
def benjamini_hochberg_decision(p_values, alpha):
    """
    Given an array of p-values for the sample from the entire model zoo,
    apply the BH procedure to decide OOD or ID.

    The steps:
        1) Sort p-values in ascending order (p(1) <= p(2) <= ... <= p(m)).
        2) threshold(j) = (j * alpha) / m, for j = 1..m.
        3) k = max { j : p(j) < threshold(j) }. If k > 0 => OOD, else => ID.

    Args:
        p_values (list or np.ndarray): p-values from all models in the zoo (m of them).
        alpha (float): BH alpha = (1 - desired TPR).

    Returns:
        is_ood (bool): True if OOD, False if ID.
    """
    import numpy as np

    m = len(p_values)
    p_values = np.array(p_values)

    # 1) Sort in ascending order
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]

    # 2) threshold(j) = (j * alpha) / m  (1-based indexing)
    thresholds = [(j+1) * alpha / m for j in range(m)]

    # 3) Find k = max j
    k = 0
    for j in range(m):
        if sorted_pvals[j] < thresholds[j]:
            k = j+1

    # If k > 0 => OOD, else => ID
    return (k > 0)


# ------------------------------------------------------------------------------
# 0b. Metric calculation: TPR, FPR, AUC
# ------------------------------------------------------------------------------
def compute_ood_metrics(labels_np, preds_np):
    """
    Computes OOD detection metrics:
    - TPR, FPR at a certain threshold (here, the threshold logic is integrated
      in the BH decision, so we effectively have a single threshold).
    - AUC for the classifier that outputs ID=1, OOD=0.

    Args:
        labels_np (np.ndarray): shape (N,), 1 = ID, 0 = OOD.
        preds_np (np.ndarray): shape (N,), 1 = predicted ID, 0 = predicted OOD.

    Returns:
        metrics (dict): {
            'accuracy': float,
            'tpr': float,
            'fpr': float,
            'auc': float
        }
    """
    from sklearn.metrics import confusion_matrix, roc_auc_score

    # We have binary predictions (ID=1, OOD=0).
    # Compute confusion matrix:
    cm = confusion_matrix(labels_np, preds_np, labels=[1, 0])
    # if labels are [1, 0], confusion matrix format:
    #    cm[0,0] = # of ID predicted as ID
    #    cm[0,1] = # of ID predicted as OOD
    #    cm[1,0] = # of OOD predicted as ID
    #    cm[1,1] = # of OOD predicted as OOD

    tp = cm[0, 0]  # ID predicted as ID
    fn = cm[0, 1]  # ID predicted as OOD
    fp = cm[1, 0]  # OOD predicted as ID
    tn = cm[1, 1]  # OOD predicted as OOD

    # TPR = tp / (tp + fn)
    if (tp + fn) == 0:
        tpr = 0.0
    else:
        tpr = tp / (tp + fn)

    # FPR = fp / (fp + tn)
    if (fp + tn) == 0:
        fpr = 0.0
    else:
        fpr = fp / (fp + tn)

    # Accuracy:
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # AUC: we need a numeric score for each sample to compute a proper ROC. 
    # But here we only have binary decisions. As a fallback, we can do:
    #    AUC ~ roc_auc_score(labels_np, preds_np)
    # though this is not a real continuous ROC curve, but a single point.
    # We'll do it to have some "AUC" in the result:
    try:
        auc_val = roc_auc_score(labels_np, preds_np)
    except ValueError:
        auc_val = 0.0  # if there's an edge case (all positives or all negatives)

    return {
        'accuracy': float(accuracy),
        'tpr': float(tpr),
        'fpr': float(fpr),
        'auc': float(auc_val)
    }


# ------------------------------------------------------------------------------
# 1. The main ZODE pipeline function
# ------------------------------------------------------------------------------
def run_zode_detection(
    model_names,
    score_method,             # "msp", "energy", "mahalanobis", or "knn"
    alpha,                    # BH alpha = (1 - target TPR)
    root_dir="datasets",
    batch_size=64,
    device="cpu",
    ood_dataset_name="svhn",  # example OOD dataset
    temperature=1.0,
    k_neighbors=1,
    class_conditional_maha=True,
    weights_dir = "None"
):
    """
    Runs the ZODE pipeline for a chosen scoring method, using the Benjamini-Hochberg
    procedure across multiple models in the zoo.

    Steps:
        1) Load ID dataset -> create model zoo.
        2) Compute ID scores array for each model (needed for p-value).
        3) Evaluate ID test set + OOD dataset -> do BH procedure for each sample.
        4) Compute TPR, FPR, AUC, etc.

    Returns:
        final_results: A dictionary containing the predictions, labels, and metrics.
    """
    # 1) Load ID dataset (e.g., CIFAR-10)
    _, _, id_test_dataset = create_id_datasets(
        dataset_name="cifar10",
        root_dir=root_dir,
        train_val_split=0.9,
        download=True
    )
    id_test_loader = DataLoader(id_test_dataset, batch_size=batch_size, shuffle=False)

    # 2) Create model zoo with pretrained weights
    if weights_dir == "None":
        zoo = create_model_zoo(model_names, device=device, num_classes=10)
    else:
        zoo = create_model_zoo(model_names, weights_dir=weights_dir, device=device, num_classes=10)

    # 3) For each model, compute ID scores
    model_id_scores_dict = {}
    model_aux_dict = {}

    for m_name, model in zoo.items():
        print(f"\n[ZODE] Preparing ID scores for model: {m_name}, method: {score_method.upper()}")

        if score_method in ["msp", "energy"]:
            id_logits = extract_id_logits(id_test_loader, model, device=device)
            id_scores_array = compute_id_scores(
                id_logits,
                score_type=score_method,
                temperature=temperature
            )
            model_id_scores_dict[m_name] = id_scores_array

        elif score_method == "mahalanobis":
            id_features = extract_id_features(id_test_loader, model, device=device)
            gauss_dict = fit_gaussian_to_id_features(
                id_features, 
                id_labels=None,  # or pass actual labels if available
                num_classes=1 if not class_conditional_maha else 10,
                shared_cov=True
            )
            id_scores_array = compute_id_scores(
                id_features,
                score_type="mahalanobis",
                gaussian_dict=gauss_dict,
                class_conditional=class_conditional_maha
            )
            model_id_scores_dict[m_name] = id_scores_array
            model_aux_dict[m_name] = gauss_dict

        elif score_method == "knn":
            id_features = extract_id_features(id_test_loader, model, device=device)
            id_scores_array = compute_id_scores(
                id_features,
                score_type="knn",
                id_feature_bank=id_features,
                k=k_neighbors
            )
            model_id_scores_dict[m_name] = id_scores_array
            model_aux_dict[m_name] = id_features
        else:
            raise ValueError(f"Unknown score method {score_method}")

        print(f"ID scores shape for {m_name}: {model_id_scores_dict[m_name].shape}")

    # 4) Load OOD dataset
    ood_dataset = create_ood_dataset(
        dataset_name=ood_dataset_name,
        root_dir=root_dir,
        split='test',
        download=True
    )
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)

    # We'll label ID=1, OOD=0
    labels_list = []
    preds_list = []

    # 5) Evaluate ID test set
    tqdm_test_loader = tqdm(id_test_loader, desc="Evaluating ID test set")
    for images, _ in tqdm_test_loader:
        images = images.to(device)
        batch_size_ = images.size(0)
        labels_list += [1]*batch_size_  # ground truth ID

        for i in range(batch_size_):
            sample_img = images[i].unsqueeze(0)
            p_values = []
            for m_name, model in zoo.items():
                score_val = None
                if score_method in ["msp", "energy"]:
                    logits = model(sample_img).squeeze(0)
                    if score_method == "msp":
                        score_val = score_msp(logits)
                        tail = "lower"  # smaller MSP => more OOD
                    else:  # energy
                        score_val = score_energy(logits, temperature=temperature)
                        tail = "upper"  # bigger energy => more OOD
                    p_val = compute_p_value(score_val, model_id_scores_dict[m_name], tail=tail)
                    p_values.append(p_val)

                elif score_method == "mahalanobis":
                    feats = _extract_single_feature(model, sample_img, device)
                    gauss_dict = model_aux_dict[m_name]
                    dist = score_mahalanobis(feats, gauss_dict, class_conditional=class_conditional_maha)
                    tail = "upper"  # bigger distance => more OOD
                    p_val = compute_p_value(dist, model_id_scores_dict[m_name], tail=tail)
                    p_values.append(p_val)

                elif score_method == "knn":
                    feats = _extract_single_feature(model, sample_img, device)
                    id_bank = model_aux_dict[m_name]
                    dist = score_knn(feats, id_bank, k=k_neighbors)
                    tail = "upper"  # bigger distance => more OOD
                    p_val = compute_p_value(dist, model_id_scores_dict[m_name], tail=tail)
                    p_values.append(p_val)

            is_ood = benjamini_hochberg_decision(p_values, alpha)
            pred_label = 0 if is_ood else 1
            preds_list.append(pred_label)

    # 6) Evaluate OOD set
    tqdm_ood_loader = tqdm(ood_loader, desc="Evaluating OOD dataset")
    for images, _ in tqdm_ood_loader:
        images = images.to(device)
        batch_size_ = images.size(0)
        labels_list += [0]*batch_size_  # ground truth OOD=0

        for i in range(batch_size_):
            sample_img = images[i].unsqueeze(0)
            p_values = []
            for m_name, model in zoo.items():
                if score_method in ["msp", "energy"]:
                    logits = model(sample_img).squeeze(0)
                    if score_method == "msp":
                        score_val = score_msp(logits)
                        tail = "lower"
                    else:
                        score_val = score_energy(logits, temperature=temperature)
                        tail = "upper"
                    p_val = compute_p_value(score_val, model_id_scores_dict[m_name], tail=tail)
                    p_values.append(p_val)

                elif score_method == "mahalanobis":
                    feats = _extract_single_feature(model, sample_img, device)
                    gauss_dict = model_aux_dict[m_name]
                    dist = score_mahalanobis(feats, gauss_dict, class_conditional=class_conditional_maha)
                    tail = "upper"
                    p_val = compute_p_value(dist, model_id_scores_dict[m_name], tail=tail)
                    p_values.append(p_val)

                elif score_method == "knn":
                    feats = _extract_single_feature(model, sample_img, device)
                    id_bank = model_aux_dict[m_name]
                    dist = score_knn(feats, id_bank, k=k_neighbors)
                    tail = "upper"
                    p_val = compute_p_value(dist, model_id_scores_dict[m_name], tail=tail)
                    p_values.append(p_val)

            is_ood = benjamini_hochberg_decision(p_values, alpha)
            pred_label = 0 if is_ood else 1
            preds_list.append(pred_label)

    # 7) Summarize results + compute metrics
    labels_np = np.array(labels_list)  # 1 => ID, 0 => OOD
    preds_np = np.array(preds_list)    # 1 => predicted ID, 0 => predicted OOD

    metrics = compute_ood_metrics(labels_np, preds_np)
    print(f"\n[ZODE] Final results for {score_method.upper()} + BH alpha={alpha}")
    print(f"  Accuracy = {metrics['accuracy']*100:.2f}%")
    print(f"  TPR      = {metrics['tpr']*100:.2f}%")
    print(f"  FPR      = {metrics['fpr']*100:.2f}%")
    print(f"  AUC      = {metrics['auc']:.4f}")

    final_results = {
        "labels": labels_np,
        "predictions": preds_np,
        "metrics": metrics
    }
    return final_results


# ------------------------------------------------------------------------------
# 2. Helper for single-sample feature extraction
# ------------------------------------------------------------------------------
def _extract_single_feature(model, image, device="cpu"):
    """
    Extract penultimate-layer features for a single image. 
    Reuses logic from scoring.py or a partial forward pass approach.
    """
    from model.model_utils import extract_penultimate_features
    feats = extract_penultimate_features(model, image, device=device)
    return feats.squeeze(0)


# ------------------------------------------------------------------------------
# 3. Main CLI
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run the ZODE pipeline using BH and compute OOD metrics.")
    parser.add_argument("--model_names", type=str, default="resnet18,densenet121",
                        help="Comma-separated model names in the zoo.")
    parser.add_argument("--score_method", type=str, default="msp",
                        choices=["msp", "energy", "mahalanobis", "knn"],
                        help="Which scoring method to use.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Alpha for BH procedure (1 - target TPR).")
    parser.add_argument("--ood_dataset", type=str, default="svhn",
                        help="Which OOD dataset to evaluate (svhn, lsun, etc.).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for energy-based scoring.")
    parser.add_argument("--k_neighbors", type=int, default=1,
                        help="K for KNN-based scoring.")
    parser.add_argument("--class_conditional_maha", action="store_true",
                        help="If set, do class-conditional Mahalanobis with multiple means/cov.")
    parser.add_argument("--weights_dir", type=str, default= "None",
                        help="Weights directory for the models.")
    
    args = parser.parse_args()
    
    model_names_list = args.model_names.split(",")

    results = run_zode_detection(
        model_names=model_names_list,
        score_method=args.score_method,
        alpha=args.alpha,
        root_dir="datasets",
        batch_size=args.batch_size,
        device=args.device,
        ood_dataset_name=args.ood_dataset,
        temperature=args.temperature,
        k_neighbors=args.k_neighbors,
        class_conditional_maha=args.class_conditional_maha,
        weights_dir=args.weights_dir
    )
    print("\nDone. Results dictionary:")
    print(results)


if __name__ == "__main__":
    main()
