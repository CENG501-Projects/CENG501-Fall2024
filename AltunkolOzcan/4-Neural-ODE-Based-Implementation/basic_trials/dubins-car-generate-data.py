import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

print(device)

true_y0 = torch.tensor([[0., 0., 0.]]).to(device)  # Initial (x, y, theta)
t = torch.linspace(0., 25., args.data_size).to(device)  # Time vector
v_max = 1.0  # Maximum velocity
r = 1.0  # Turning radius

class Lambda(nn.Module):
    def forward(self, t, y):
        # Generate random control inputs for training data
        alpha_t = torch.rand(1, device=y.device) * 2 - 1  # Uniformly sampled in [-1, 1]
        # alpha_t = torch.tensor([0.]).to(device)
        v_t = torch.rand(1, device=y.device) * v_max      # Uniformly sampled in [0, v_max]

        # Dubins car dynamics
        x, y_pos, theta = y[:, 0], y[:, 1], y[:, 2]
        dxdt = v_t * torch.cos(theta)
        dydt = v_t * torch.sin(theta)
        dthetadt = alpha_t * v_t / r

        # Return the combined derivatives
        return torch.stack((dxdt, dydt, dthetadt), dim=1)

with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='euler')

print(true_y)   

