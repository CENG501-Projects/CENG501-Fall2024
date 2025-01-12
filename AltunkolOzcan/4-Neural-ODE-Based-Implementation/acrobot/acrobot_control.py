import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ExponentialLR
from util.Sin import Sin
from .acrobotDynamics import acrobotDynamics
from .acrobot_dataset import sampleYdotAcrobot

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams','rk4'], default='rk4')
parser.add_argument('--time_horizon', type=float, default=128)
parser.add_argument('--data_size', type=int, default=8192)
parser.add_argument('--batch_time', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

print(device)

class AcrobotController(nn.Module):
    def __init__(self):
        super(AcrobotController, self).__init__()
        self.u_lim = 250
        self.net = nn.Sequential(
            nn.Linear(4, 16),  # Current state
            Sin(),
            nn.Linear(16,64),
            nn.LeakyReLU(), 
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 8),
            Sin(),
            nn.Linear(8, 1)
        )
        self.tanh = nn.Tanh()

        # Weight initialization for the layers
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self,x):
        control_input = self.net(x)  # Compute the output of the network
        scaled_output = self.u_lim * self.tanh(control_input)  # Scale the Tanh output
        return scaled_output
