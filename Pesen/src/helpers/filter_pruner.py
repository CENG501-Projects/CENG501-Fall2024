import torch


class FilterPruner:
    def __init__(self, layers_afie, num_filters_per_layer):
        """
        Initialize the FilterPruner with AFIE scores and filter counts.

        Args:
            layers_afie (list): List of AFIE scores for each layer.
            num_filters_per_layer (list): Number of filters in each layer.
        """
        self.layers_afie = layers_afie
        self.num_filters_per_layer = num_filters_per_layer

    def compute_pruning_ratios(self, overall_pruning_ratio=0.5, lambda_min=0.1):
        """
        Compute layer-wise pruning ratios based on AFIE scores.

        Args:
            overall_pruning_ratio (float): Desired overall pruning ratio.
            lambda_min (float): Minimum pruning threshold.

        Returns:
            pruning_ratios (list): List of pruning ratios for each layer.
        """
        afie_max = max(self.layers_afie)

        # Ensure no zero AFIE scores
        sanitized_afie_scores = [max(afie, 1e-8) for afie in self.layers_afie]  # Replace 0 with a small value

        # Calculate initial pruning ratios for each layer
        pruning_ratios = [
            max(0.01, min(1.0, lambda_min * (afie_max / afie)))  # Enforce a minimum and maximum pruning ratio
            for afie in sanitized_afie_scores
        ]

        total_filters = sum(self.num_filters_per_layer)
        total_pruned_filters = overall_pruning_ratio * total_filters

        # Normalize pruning ratios to meet the overall pruning constraint
        scaling_factor = total_pruned_filters / sum(
            pruning_ratios[i] * self.num_filters_per_layer[i]
            for i in range(len(self.layers_afie))
        )
        pruning_ratios = [min(1.0, ratio * scaling_factor) for ratio in pruning_ratios]

        return pruning_ratios


    def prune_filters(self, weight_matrices, pruning_ratios):
        """
        Prune filters from each layer based on pruning ratios.

        Args:
            weight_matrices (list): List of weight matrices for each layer.
            pruning_ratios (list): List of pruning ratios for each layer.

        Returns:
            pruned_weights (list): List of pruned weight matrices.
        """
        pruned_weights = []
        for weights, ratio in zip(weight_matrices, pruning_ratios):
            num_filters_to_keep = max(1, int(weights.size(0) * (1 - ratio)))  # Ensure at least 1 filter is retained
            eigenvalues = torch.linalg.svdvals(weights.view(weights.size(0), -1))**2
            top_filters = torch.argsort(eigenvalues, descending=True)[:num_filters_to_keep]
            pruned_weights.append(weights[top_filters])
        return pruned_weights

