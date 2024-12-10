import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisionLoss(nn.Module):
    def __init__(self):
        super(SupervisionLoss, self).__init__()

    def forward(self, coarse_probs, gt_matches, gt_unmatched, fine_positions, gt_positions, view_switch_logits, gt_switch_labels):
        """
        Compute the overall supervision loss.

        Args:
            coarse_probs: Tensor, shape (B, N, Hc * Wc + 1), coarse matching probabilities.
            gt_matches: List of Tensors, ground-truth matches [(i, j) indices per batch].
            gt_unmatched: List of Tensors, unmatched indices for each batch.
            fine_positions: Tensor, shape (B, K, 2), refined positions.
            gt_positions: Tensor, shape (B, K, 2), ground-truth fine positions.
            view_switch_logits: Tensor, shape (B, 2), logits for view switcher.
            gt_switch_labels: Tensor, shape (B,), binary labels for view switcher.

        Returns:
            total_loss: Tensor, scalar, combined loss.
        """
        batch_size = coarse_probs.shape[0]
        total_loss = 0

        # Coarse Matching Loss (Lm_c)
        coarse_match_loss = 0
        for b in range(batch_size):
            match_indices = gt_matches[b]
            if len(match_indices) > 0:
                match_probs = coarse_probs[b][match_indices[:, 0], match_indices[:, 1]]
                coarse_match_loss -= torch.log(match_probs).mean()
        coarse_match_loss /= batch_size

        # Dustbin Loss (Ld_c)
        dustbin_loss = 0
        for b in range(batch_size):
            unmatched_indices = gt_unmatched[b]
            if len(unmatched_indices) > 0:
                dustbin_probs = coarse_probs[b][unmatched_indices, -1]
                dustbin_loss -= torch.log(dustbin_probs).mean()
        dustbin_loss /= batch_size

        # Fine Matching Loss (Lf)
        fine_match_loss = F.mse_loss(fine_positions, gt_positions)

        # View Switcher Loss (Lvs)
        view_switch_loss = F.cross_entropy(view_switch_logits, gt_switch_labels)

        # Combine Losses
        total_loss = coarse_match_loss + dustbin_loss + fine_match_loss + view_switch_loss
        return total_loss
