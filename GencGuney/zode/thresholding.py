import numpy as np

def compute_empirical_cdf(score, validation_scores):
    """
    Compute the empirical cumulative distribution function (CDF) value for a score.

    Args:
        score (float): The detection score of the test sample.
        validation_scores (list or np.ndarray): Scores from the validation set (in-distribution data).

    Returns:
        float: The empirical CDF value, interpreted as a p-value.
    """
    validation_scores = np.array(validation_scores)
    empirical_cdf = np.sum(validation_scores <= score) / len(validation_scores)
    return empirical_cdf


def compute_p_values(test_scores, validation_scores_per_model):
    """
    Compute p-values for a list of test scores using validation data from each model.

    Args:
        test_scores (list): Detection scores for the test sample, one for each model.
        validation_scores_per_model (list of list): Validation scores for each model.

    Returns:
        list: p-values corresponding to the test scores.
    """
    p_values = []
    for score, validation_scores in zip(test_scores, validation_scores_per_model):
        p_value = compute_empirical_cdf(score, validation_scores)
        p_values.append(p_value)
    return p_values


def apply_benjamini_hochberg(p_values, alpha):
    """
    Apply the Benjamini-Hochberg procedure to adjust p-value thresholds.

    Args:
        p_values (list or np.ndarray): List of p-values to process.
        alpha (float): Target false discovery rate (FDR).

    Returns:
        int: Largest index k satisfying the threshold condition.
    """
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.array(p_values)[sorted_indices]
    thresholds = np.arange(1, m + 1) / m * alpha

    # Find the largest k such that p(k) <= k/m * alpha
    k = np.max(np.where(sorted_p_values <= thresholds)[0]) if np.any(sorted_p_values <= thresholds) else -1
    return k, sorted_indices


def get_selected_models(p_values, sorted_indices, k, model_zoo):
    """
    Retrieve the selected models based on the Benjamini-Hochberg correction results.

    Args:
        p_values (list): Original list of p-values.
        sorted_indices (np.ndarray): Indices of p-values sorted in ascending order.
        k (int): Largest index satisfying the BH condition.
        model_zoo (list): List of models in the zoo.

    Returns:
        list: Selected models if k >= 0, else an empty list.
    """
    if k == -1:
        return []  # No models selected
    selected_indices = sorted_indices[:k + 1]
    selected_models = [model_zoo[i] for i in selected_indices]
    return selected_models
