import torch.nn as nn
import torch.nn.functional as F
import torch
class CombinedLoss(nn.Module):
    def __init__(self, lambda_weight):
        super(CombinedLoss, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, factual_fused, counterfactual_fused, predictions, labels):

        """print("FACT: ", factual_fused)
        print("COUNTER: ", counterfactual_fused)"""
        factual_pred_loss = F.cross_entropy(factual_fused, labels)
        counterfactual_pred_loss = F.cross_entropy(counterfactual_fused, labels)

        p_log = torch.log(counterfactual_fused)
        bias_loss = F.kl_div(p_log, factual_fused, reduction='batchmean')


        total_loss = factual_pred_loss + counterfactual_pred_loss + self.lambda_weight * bias_loss
        """print("FACTUAL LOSS:", factual_pred_loss)
        print("COUNTER LOSS: ", counterfactual_pred_loss)
        print("BIAS LOSS: ", bias_loss)
        print("TOTAL LOSS: ", total_loss)"""
        return total_loss









