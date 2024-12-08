import numpy as np
from scipy.stats import norm

def compute_p_value(scores, validation_scores):
    """
    Compute p-values based on the detection scores and validation data.
    
    Args:
        scores (float): The detection score of the test sample.
        validation_scores (list): Scores from the validation set (in-distribution data).

    Returns:
        float: p-value for the test sample.
    """
    empirical_cdf = np.sum(validation_scores <= scores) / len(validation_scores)
    return empirical_cdf


def benjamini_hochberg_correction(p_values, alpha):
    """
    Apply the Benjamini-Hochberg procedure to adjust p-value thresholds.
    
    Args:
        p_values (list): List of p-values.
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
    return k


def zode_algorithm(test_sample, model_zoo, validation_set, score_function, alpha=0.05):
    """
    Implementation of the ZODE algorithm for OOD detection.
    
    Args:
        test_sample (array-like): The input test sample.
        model_zoo (list): List of pre-trained models in the zoo.
        validation_set (list): Validation dataset of in-distribution samples.
        score_function (callable): Function to compute detection scores.
        alpha (float): Target true positive rate (TPR).

    Returns:
        dict: Contains the classification ('ID' or 'OOD') and selected models.
    """
    p_values = []
    scores = []

    # Compute scores and p-values for each model in the zoo
    for model in model_zoo:
        # Compute score for the test sample
        test_score = score_function(test_sample, model)
        scores.append(test_score)

        # Compute p-value using validation data
        validation_scores = [score_function(sample, model) for sample in validation_set]
        p_value = compute_p_value(test_score, np.array(validation_scores))
        p_values.append(p_value)

    # Sort p-values and apply Benjamini-Hochberg correction
    k = benjamini_hochberg_correction(p_values, alpha)

    # Determine classification and selected models
    if k == -1:
        # No models pass the threshold, classify as ID
        result = {"classification": "ID", "selected_models": []}
    else:
        # Select the top k models
        selected_models = [model_zoo[i] for i in np.argsort(p_values)[:k + 1]]
        result = {"classification": "OOD", "selected_models": selected_models}

    return result
