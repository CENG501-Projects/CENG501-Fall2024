import torch
import torch.nn.functional as F

class LayerPruningAnalyzer:
    def __init__(self, weight_matrix):
        """
        Initialize the LayerPruningAnalyzer with a weight matrix.

        Args:
            weight_matrix (torch.Tensor): The 2D weight matrix of a convolutional layer.
        """
        self.weight_matrix = weight_matrix

    def compute_entropy(self, eigenvalues):
        """
        Compute the information entropy (K_l) for a layer.

        Args:
            eigenvalues (torch.Tensor): Eigenvalues of the weight matrix (1D tensor).

        Returns:
            entropy (float): The computed information entropy (K_l).
        """
        # Step 1: Softmax normalization of eigenvalues
        probabilities = F.softmax(eigenvalues, dim=0)

        # Step 2: Compute entropy using the formula K_l = -sum(s_l^(i,soft) * log(s_l^(i,soft)))
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10))  # Add epsilon to avoid log(0)
        return entropy.item()

    def compute_afie(self, eigenvalues):
        """
        Compute the Average Filter Information Entropy (AFIE) for a layer.

        Args:
            eigenvalues (torch.Tensor): Eigenvalues of the weight matrix (1D tensor).

        Returns:
            afie (float): The computed AFIE score for the layer.
        """
        # Compute entropy (K_l)
        entropy = self.compute_entropy(eigenvalues)

        # Normalize entropy by the number of filters (c_l) to calculate AFIE
        num_filters = eigenvalues.size(0)
        afie = entropy / num_filters
        return afie

    def compute_afie_and_pruning_ratio(self, overall_pruning_ratio=0.5):
        """
        Compute the eigenvalue distribution, AFIE score, and determine the pruning ratio for the layer.

        Args:
            overall_pruning_ratio (float): The desired overall pruning ratio for the model.

        Returns:
            afie (float): The AFIE score for the layer.
            pruning_ratio (float): The calculated pruning ratio for the layer.
        """
        # Step 1: Decompose weight matrix using Singular Value Decomposition (SVD)
        U, S, V = torch.svd(self.weight_matrix)  # SVD produces singular values
        eigenvalues = S**2  # Compute eigenvalues as squared singular values

        # Step 2: Compute AFIE
        afie = self.compute_afie(eigenvalues)

        # Step 3: Compute Pruning Ratio
        pruning_ratio = overall_pruning_ratio * (1 - (afie / eigenvalues.numel()))  # Example scaling
        return afie, pruning_ratio

    def analyze_redundancy(self, threshold=0.01):
        """
        Analyze redundancy in the weight matrix by determining significant eigenvalues.

        Args:
            threshold (float): Threshold to determine significant eigenvalues (default: 0.01).

        Returns:
            rank (int): The effective rank of the weight matrix based on the eigenvalue threshold.
            keep_filters (list): Indices of filters to retain.
            prune_filters (list): Indices of filters to prune.
        """
        # Step 1: Perform SVD to get eigenvalues
        U, S, V = torch.svd(self.weight_matrix)
        eigenvalues = S**2  # Compute eigenvalues as squared singular values

        # Step 2: Compute total "intensity" of eigenvalues
        total_intensity = torch .sum(eigenvalues).item()

        # Step 3: Identify significant eigenvalues
        significant_eigenvalues = eigenvalues[eigenvalues / total_intensity > threshold]
        rank = len(significant_eigenvalues)  # Rank of the weight matrix

        # Step 4: Decide which filters to keep
        keep_filters = torch.argsort(eigenvalues, descending=True)[:rank]  # Top-ranked filters
        prune_filters = torch.argsort(eigenvalues, descending=True)[rank:]  # Low-ranked filters

        return rank, keep_filters.tolist(), prune_filters.tolist()
