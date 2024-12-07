import numpy as np
from typing import Any, List, Tuple

class AreaOverPerturbationCurve:
    """
    Area Over the Perturbation Curve (AOPC) metric.
    Measures the average change in output probability when gradually removing
    the most important features identified by the attribution method.
    """
    
    def __init__(self, num_steps: int = 5):
        self.num_steps = num_steps
    
    def compute(self, model: Any, input_data: np.ndarray, attribution_scores: np.ndarray) -> float:
        """
        Compute AOPC score.
        
        Args:
            model: The model to evaluate
            input_data: Original input data
            attribution_scores: Importance scores for each feature
            
        Returns:
            float: AOPC score
        """
        original_pred = self._get_prediction(model, input_data)
        sorted_indices = np.argsort(attribution_scores)[::-1]
        
        step_size = len(sorted_indices) // self.num_steps
        aopc_score = 0.0
        
        for step in range(self.num_steps):
            perturbed_input = input_data.copy()
            indices_to_remove = sorted_indices[:(step + 1) * step_size]
            perturbed_input[indices_to_remove] = 0  # or other perturbation strategy
            
            new_pred = self._get_prediction(model, perturbed_input)
            aopc_score += (original_pred - new_pred)
            
        return aopc_score / self.num_steps
    
    def _get_prediction(self, model: Any, input_data: np.ndarray) -> float:
        """Get model prediction probability for the target class."""
        return model(input_data) 