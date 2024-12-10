import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ExponentialLR
from Sin import Sin

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--time_horizon', type=float, default=400)
parser.add_argument('--data_size', type=int, default=32000)
parser.add_argument('--batch_time', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=200)
parser.add_argument('--epochs', type=int, default=50)
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
t = torch.linspace(0., args.time_horizon, args.data_size).to(device)  # Time vector
v_max = 1.0  # Maximum velocity
r = 1.0  # Turning radius

# Generate random control inputs for training data
alpha_t_seq = torch.rand(args.data_size, device=true_y0.device) * 2 - 1  # Uniformly sampled in [-1, 1]
v_t_seq = torch.rand(args.data_size, device=true_y0.device) * v_max      # Uniformly sampled in [0, v_max]
true_u = torch.stack((v_t_seq, alpha_t_seq), dim=1)  # Control inputs
true_u = true_u.view(args.data_size, 1, 2)  # or tensor.reshape(1000, 1, 2)

class Lambda(nn.Module):
    def __init__(self, alpha_t_seq, v_t_seq):
        super(Lambda, self).__init__()
        self.alpha_t_seq = alpha_t_seq  # Store control inputs internally
        self.v_t_seq = v_t_seq

    def forward(self, t, y):
        # Find the index of the current time step
        time_index = int(t * (args.data_size - 1) / args.time_horizon)  # Map t to an index
        # Select the control inputs for the current time step
        alpha_t = alpha_t_seq[time_index]
        v_t = v_t_seq[time_index]

        # Dubins car dynamics
        theta = y[:, 2]
        dxdt = v_t * torch.cos(theta)
        dydt = v_t * torch.sin(theta)
        dthetadt = torch.tensor([alpha_t * v_t / r])
        return torch.stack((dxdt, dydt, dthetadt), dim=1)

# Create the ODE function instance
lambda_model = Lambda(alpha_t_seq, v_t_seq)

with torch.no_grad():
    true_y = odeint(lambda_model, true_y0, t, method='euler')

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)

    batch_u = torch.stack([true_u[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, 2)

    return batch_y0.to(device), batch_t.to(device), batch_y.to(device), batch_u.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

args.viz = 1

if args.viz:
    """makedirs('png')"""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-200, 200)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-200, 200)
        ax_phase.set_ylim(-200, 200)
        """
        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)
        """
        fig.tight_layout()
        """plt.savefig('png/{:03d}'.format(itr))"""
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):
    def __init__(self, batch_u=None):
        super(ODEFunc, self).__init__()
        # Control input initialization (optional at this stage)
        self.batch_u = batch_u  # Control input batch (alpha_t, v_t)
        
        self.net = nn.Sequential(
            nn.Linear(5, 50),  # 2 states (x, y, theta) + 2 control inputs (alpha_t, v_t) = 4 inputs
            Sin(),
            nn.Linear(50, 20),
            Sin(),
            nn.Linear(20, 3),  # Output for the derivatives (dx/dt, dy/dt, dtheta/dt)
        )

        # Weight initialization for the layers
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # Find the index of the current time step
        time_index = int((t * (self.batch_u.shape[0] - 1)) / args.time_horizon)  # Map t to an index
        # Select the control inputs for the current time step
        u = self.batch_u[time_index]

        # Concatenate control input with the state
        # Here, y is the state, and u is the control input
        batch_input = torch.cat([y, u], dim= -1)

        # Pass concatenated state and control input to the neural network
        return self.net(batch_input)

    def update_control_input(self, new_control_input):
        # Method to update the control input for each new batch
        self.batch_u = new_control_input


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0
    
    func = ODEFunc().to(device)
    optimizer = optim.Adam(func.parameters(), lr=3e-3)
    scheduler = ExponentialLR(optimizer, gamma=0.9)  # Decays LR by 10% every epoch
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    
    for epoch in range(1, args.epochs + 1):
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y, batch_u = get_batch()
            
            # Update the control input in the ODE function with the new batch
            func.update_control_input(batch_u)  # Update control input

            pred_y = odeint(func, batch_y0, batch_t, method='euler').to(device)
            loss = torch.mean(torch.abs(pred_y - batch_y))
            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - end)
            loss_meter.update(loss.item())

            if itr % args.test_freq == 0:
                with torch.no_grad():
                    # Update the control input in the ODE function with the new batch
                    func.update_control_input(true_u)  # Update control input
                    pred_y = odeint(func, true_y0, t, method='euler')
                    loss = torch.mean(torch.abs(pred_y - true_y))
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                    visualize(true_y, pred_y, func, ii)
                    ii += 1
        end = time.time()
        scheduler.step()  # Decay the learning rate at the end of each epoch
        
    
    """Export the obtained model parameters"""
    torch.save(func.state_dict(), 'dubins.pth')
    print('Model parameters exported to dubins.pth')

    """
    Use the following code to load the model parameters from file
    func = ODEFunc().to(device)
    func.load_state_dict(torch.load('dubins.pth'))
    
    Given the initial state, the following code can be used to predict the trajectory
    with torch.no_grad():
        pred_y = odeint(func, true_y0, t)

    and visualize the trajectory using the visualize function
    visualize(true_y, pred_y, func, 0)
    
    """
    

