import numpy as np
from typing import Any, List
from scipy.special import logit

class LogOddsShift:
    """
    Log-Odds Shift metric.
    Measures the change in log-odds of the model prediction when important
    features are removed.
    """
    
    def compute(self, model: Any, input_data: np.ndarray, 
                attribution_scores: np.ndarray, target_class: int) -> float:
        """
        Compute log-odds shift.
        
        Args:
            model: The model to evaluate
            input_data: Original input data
            attribution_scores: Importance scores for each feature
            target_class: Target class index
            
        Returns:
            float: Log-odds shift score
        """
        original_prob = self._get_prediction(model, input_data, target_class)
        original_logit = logit(np.clip(original_prob, 1e-7, 1-1e-7))
        
        perturbed_input = self._remove_important_features(input_data, attribution_scores)
        perturbed_prob = self._get_prediction(model, perturbed_input, target_class)
        perturbed_logit = logit(np.clip(perturbed_prob, 1e-7, 1-1e-7))
        
        return original_logit - perturbed_logit
    
    def _get_prediction(self, model: Any, input_data: np.ndarray, 
                       target_class: int) -> float:
        """Get model prediction probability for the target class."""
        return model(input_data)[target_class]
    
    def _remove_important_features(self, input_data: np.ndarray, 
                                 attribution_scores: np.ndarray) -> np.ndarray:
        """Remove features with highest attribution scores."""
        threshold = np.percentile(attribution_scores, 90)
        mask = attribution_scores >= threshold
        perturbed_input = input_data.copy()
        perturbed_input[mask] = 0
        return perturbed_input 