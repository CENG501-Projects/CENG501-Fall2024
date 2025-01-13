import torch.nn as nn
import torch.nn.functional as F
import torch

class DiscreteLoss(nn.Module):
    ''' Class to measure loss between categorical emotion predictions and labels.'''

    def __init__(self, weight_type='mean', device=torch.device('cuda')):
        super(DiscreteLoss, self).__init__()
        self.weight_type = weight_type
        self.device = device
        if self.weight_type == 'mean':
            self.weights = torch.ones((1, 26)) / 26.0
            self.weights = self.weights.to(self.device)
        elif self.weight_type == 'static':
            self.weights = torch.FloatTensor([0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
                                              0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
                                              0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520,
                                              0.1537]).unsqueeze(0)
            self.weights = self.weights.to(self.device)

    def forward(self, pred, target):
        if self.weight_type == 'dynamic':
            self.weights = self.prepare_dynamic_weights(target)
            self.weights = self.weights.to("cuda")
        loss = (((pred - target) ** 2) * self.weights)
        return loss.sum()

    def prepare_dynamic_weights(self, target):
        target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
        weights = torch.zeros((1, 26))
        weights[target_stats != 0] = 1.0 / torch.log(target_stats[target_stats != 0].data + 1.2)
        weights[target_stats == 0] = 0.0001
        return weights


class CombinedLossMultiLabel(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(CombinedLossMultiLabel, self).__init__()
        self.device = device
        self.discrete_loss = DiscreteLoss(weight_type='dynamic', device=device)

    def forward(self, factual_fused, counterfactual_fused, predictions, labels):

        factual_pred_loss = self.discrete_loss(factual_fused, labels)
        counterfactual_pred_loss = self.discrete_loss(counterfactual_fused, labels)

        p_log = torch.log(counterfactual_fused)
        bias_loss = F.kl_div(p_log, factual_fused, reduction='batchmean')

        # Total loss
        total_loss = factual_pred_loss + counterfactual_pred_loss + bias_loss

        return total_loss

class CombinedLoss(nn.Module):
    def __init__(self,):
        super(CombinedLoss, self).__init__()

    def forward(self, factual_fused, counterfactual_fused, predictions, labels):

        factual_pred_loss = F.cross_entropy(factual_fused, labels)
        counterfactual_pred_loss = F.cross_entropy(counterfactual_fused, labels)

        p_log = torch.log(counterfactual_fused)
        bias_loss = F.kl_div(p_log, factual_fused, reduction='batchmean')


        total_loss = factual_pred_loss + counterfactual_pred_loss +  bias_loss
        return total_loss









