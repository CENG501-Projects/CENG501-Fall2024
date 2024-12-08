import numpy as np
from typing import Any, List

class Sufficiency:
    """
    Sufficiency metric.
    Measures how model predictions change when only the most important features
    are kept in the input.
    """
    
    def __init__(self, keep_ratio: float = 0.1):
        self.keep_ratio = keep_ratio
    
    def compute(self, model: Any, input_data: np.ndarray, 
                attribution_scores: np.ndarray) -> float:
        """
        Compute sufficiency score.
        
        Args:
            model: The model to evaluate
            input_data: Original input data
            attribution_scores: Importance scores for each feature
            
        Returns:
            float: Sufficiency score
        """
        original_pred = self._get_prediction(model, input_data)
        
        num_features_to_keep = int(len(attribution_scores) * self.keep_ratio)
        top_indices = np.argsort(attribution_scores)[-num_features_to_keep:]
        
        sufficient_input = np.zeros_like(input_data)
        sufficient_input[top_indices] = input_data[top_indices]
        
        sufficient_pred = self._get_prediction(model, sufficient_input)
        
        return original_pred - sufficient_pred
    
    def _get_prediction(self, model: Any, input_data: np.ndarray) -> float:
        """Get model prediction probability for the target class."""
        return model(input_data) 