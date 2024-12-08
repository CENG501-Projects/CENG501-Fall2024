import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def compute_tpr_fpr(predictions, ground_truth, threshold=0.5):
    """
    Compute True Positive Rate (TPR) and False Positive Rate (FPR).
    
    Args:
        predictions (list or np.ndarray): Predicted scores (higher for ID samples).
        ground_truth (list or np.ndarray): Binary labels (1 for ID, 0 for OOD).
        threshold (float): Decision threshold.

    Returns:
        tuple: TPR and FPR values.
    """
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Binary classification based on the threshold
    predicted_labels = (predictions >= threshold).astype(int)

    # Compute TPR and FPR
    true_positives = np.sum((predicted_labels == 1) & (ground_truth == 1))
    false_positives = np.sum((predicted_labels == 1) & (ground_truth == 0))
    true_negatives = np.sum((predicted_labels == 0) & (ground_truth == 0))
    false_negatives = np.sum((predicted_labels == 0) & (ground_truth == 1))

    tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

    return tpr, fpr


def compute_auc(predictions, ground_truth):
    """
    Compute Area Under the Curve (AUC) for ROC.
    
    Args:
        predictions (list or np.ndarray): Predicted scores (higher for ID samples).
        ground_truth (list or np.ndarray): Binary labels (1 for ID, 0 for OOD).

    Returns:
        float: AUC value.
    """
    return roc_auc_score(ground_truth, predictions)


def evaluate_model(test_scores, test_labels, target_tpr=0.95):
    """
    Evaluate the performance of a model on OOD detection using TPR, FPR, and AUC.
    
    Args:
        test_scores (list or np.ndarray): Predicted scores for test samples.
        test_labels (list or np.ndarray): Binary ground truth labels (1 for ID, 0 for OOD).
        target_tpr (float): Target TPR for calculating corresponding FPR.

    Returns:
        dict: Evaluation metrics (TPR, FPR, and AUC).
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(test_labels, test_scores)

    # Find the threshold corresponding to the target TPR
    threshold_idx = np.argmin(np.abs(tpr - target_tpr))
    fpr_at_tpr = fpr[threshold_idx]

    # Compute AUC
    auc = compute_auc(test_scores, test_labels)

    return {
        "TPR": tpr[threshold_idx],
        "FPR": fpr_at_tpr,
        "AUC": auc,
        "Threshold": thresholds[threshold_idx],
    }


def evaluate_zode_results(results, test_labels):
    """
    Evaluate ZODE results by computing TPR, FPR, and AUC.

    Args:
        results (list): Classification results from ZODE (e.g., ['ID', 'OOD', ...]).
        test_labels (list or np.ndarray): Binary ground truth labels (1 for ID, 0 for OOD).

    Returns:
        dict: Evaluation metrics for ZODE results.
    """
    binary_predictions = [1 if result == "ID" else 0 for result in results]
    test_labels = np.array(test_labels)

    # Compute metrics
    tpr, fpr = compute_tpr_fpr(binary_predictions, test_labels)
    auc = compute_auc(binary_predictions, test_labels)

    return {
        "TPR": tpr,
        "FPR": fpr,
        "AUC": auc,
    }
