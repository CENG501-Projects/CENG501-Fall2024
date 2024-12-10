import torch
import torch.nn as nn

class Sin(nn.Module):
  def forward(self, x):
    return torch.sin(x)