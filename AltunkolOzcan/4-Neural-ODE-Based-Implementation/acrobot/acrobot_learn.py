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
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--time_horizon', type=float, default=128)
parser.add_argument('--data_size', type=int, default=8192)
parser.add_argument('--batch_time', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=15)
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

true_y0 = torch.zeros((4,)).to(device)  # Initial (x, y, theta)

t = torch.linspace(0., args.time_horizon, args.data_size).to(device)  # Time vector
# define input bounds
u_max = 1
u_min = -1

# Generate random control inputs for training data
u_seq = u_min + (u_max - u_min)*torch.rand(args.data_size, device=device)
true_u = u_seq
# true_u = torch.stack((v_t_seq, alpha_t_seq), dim=1)  # Control inputs
# true_u = true_u.view(args.data_size, 1, 2)  # or tensor.reshape(1000, 1, 2)

class Lambda(nn.Module):
    def __init__(self, u_seq, p):
        super(Lambda, self).__init__()
        self.u = u_seq
        self.p = p

    def forward(self, t, y):
        # Find the index of the current time step
        time_index = int(t * (args.data_size - 1) / args.time_horizon)  # Map t to an index
        return acrobotDynamics(y,self.u[time_index],self.p)

# Create the ODE function instance
p = {
    'm1': 10,
    'm2': 20,
    'g': 9,
    'l1': 3,
    'l2': 5
}
lambda_model = Lambda(u_seq, p)

with torch.no_grad():
    #true_y = odeint(lambda_model, true_y0, t, method='euler')
    true_y, true_ydot = sampleYdotAcrobot(true_u,p, args.data_size)
    true_traj_y = odeint(lambda_model, true_y0, t, method="rk4")

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    batch_ydot = torch.stack([true_ydot[s + i] for i in range(args.batch_time)], dim=0)
    batch_u = torch.stack([true_u[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, 1)
    batch_ytraj = torch.stack([true_traj_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    # print("Batch_y:{:.4f}", batch_y)
    # print("Batch ydot: {:.4f}", batch_ydot)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device), batch_u.to(device), batch_ydot.to(device), batch_ytraj.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if args.viz:
    """makedirs('png')"""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    ax_traj = fig.add_subplot(121, frameon=False)
    ax_traj2 = fig.add_subplot(122, frameon=False)
    ax_traj3 = fig.add_subplot(221, frameon=False)
    ax_traj4 = fig.add_subplot(222, frameon=False)
    # fig_dot = plt.figure(figsize=(8,8), facecolor="white")
    # ax = fig_dot.add_subplot(111, frameon=False)
    plt.show(block=False)

def visualizeLearnedDots(true_ydot, pred_ydot, iteration):
    """
    Plots true_ydot and pred_ydot on the same plot.
    Dynamically updates the plot as the network learns.
    """
    ax.cla()  # Clear the current axes for dynamic updating
    ax.set_title(f'Iteration {iteration}', fontsize=14)
    ax.set_xlabel('State Dimension', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    
    # Assume true_ydot and pred_ydot are tensors of shape (batch_size, state_dim)
    true_ydot = true_ydot.cpu().numpy()
    pred_ydot = pred_ydot.cpu().numpy()
    
    # Plot true_ydot
    for i in range(true_ydot.shape[1]):
        ax.plot(true_ydot[:, i], label=f'True ydot Dim {i}', linestyle='-', marker='o')

    # Plot pred_ydot
    for i in range(pred_ydot.shape[1]):
        ax.plot(pred_ydot[:, i], label=f'Pred ydot Dim {i}', linestyle='--')

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True)

    plt.pause(0.001)  # Pause to update the plot dynamically
    plt.draw()


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0,], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0,], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-200, 200)
        ax_traj.legend()

        ax_traj2.cla()
        ax_traj2.set_title('Trajectories')
        ax_traj2.set_xlabel('t')
        ax_traj2.set_ylabel('x,y')
        ax_traj2.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 1,], 'g-')
        ax_traj2.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 1,], 'b--')
        ax_traj2.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj2.set_ylim(-200, 200)
        ax_traj2.legend()

        ax_traj3.cla()
        ax_traj3.set_title('Trajectories')
        ax_traj3.set_xlabel('t')
        ax_traj3.set_ylabel('x,y')
        ax_traj3.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 2,], 'g-')
        ax_traj3.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 2,], 'b--')
        ax_traj3.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj3.set_ylim(-200, 200)
        ax_traj3.legend()

        ax_traj4.cla()
        ax_traj4.set_title('Trajectories')
        ax_traj4.set_xlabel('t')
        ax_traj4.set_ylabel('x,y')
        ax_traj4.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 3,], 'g-')
        ax_traj4.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 3,], 'b--')
        ax_traj4.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj4.set_ylim(-200, 200)
        ax_traj4.legend()
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
            nn.Linear(5, 32),  # 4 states  + 1 control input
            Sin(),
            nn.Linear(32, 32),
            Sin(),
            nn.Linear(32, 10),
            Sin(),
            nn.Linear(10, 4),  # Output for the derivatives (dx/dt, dy/dt, dtheta/dt)
        )

        # Weight initialization for the layers
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                #nn.init.normal_(m.weight, mean=0, std=0.1)
                #nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        
        # Find the index of the current time step
        if t.numel()>1:
            time_index = ((t * (self.batch_u.shape[0] - 1)) / args.time_horizon).long()  # Map t to an index
        else:
            time_index = int((t.item() * (self.batch_u.shape[0] - 1)) / args.time_horizon)

        # Select the control inputs for the current time step
        u = self.batch_u[time_index]
        u = u.unsqueeze(-1)
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

    #visualizeLearnedDots(true_ydot, true_y, 1)
    ii = 0
    
    func = ODEFunc().to(device)
    optimizer = optim.Adam(func.parameters(), lr=2e-2)
    scheduler = ExponentialLR(optimizer, gamma=0.85)  # Decays LR by 10% every epoch
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    
    loss_history = []
    valid_loss_history = []
    for epoch in range(1, args.epochs + 1):
        for itr in range(1, args.niters + 1):
            # print(true_y0.shape)
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y, batch_u, batch_ydot, batch_ytraj = get_batch()
            
            # Update the control input in the ODE function with the new batch
            func.update_control_input(batch_u)  # Update control input
            #pred_y = odeint(func, batch_y0, batch_t, method='euler').to(device)
            #loss = torch.mean(torch.abs(pred_y - batch_y))
            pred_ydot = func.forward(batch_t, batch_y)
            loss = torch.mean((pred_ydot - batch_ydot)**2)
            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - end)
            loss_meter.update(loss.item())

            if itr % args.test_freq == 0:
                with torch.no_grad():
                    # Update the control input in the ODE function with the new batch
                    func.update_control_input(true_u)  # Update control input
                    pred_ytraj = odeint(func, true_y0, t, method='rk4')
                    #loss = torch.mean(torch.abs(pred_ytraj - true_traj_y))
                    pred_ydot = func.forward(t,true_y)
                    valid_loss = torch.mean((pred_ydot - true_ydot)**2)
                    valid_loss_history.append(valid_loss)
                    print('Epoch: {:4d} | Iter {:04d} | Total Loss {:.6f}'.format(epoch, itr, loss.item()))
                    #visualizeLearnedDots(true_ydot, pred_ydot,ii)
                    visualize(true_traj_y, pred_ytraj, func, ii)
                    ii += 1
        end = time.time()
        scheduler.step()  # Decay the learning rate at the end of each epoch
        torch.save(func.state_dict(), "acrobot_1.pth")
        loss_history.append(loss_meter.avg)
    
    """Export the obtained model parameters"""
    #torch.save(func.state_dict(), 'dubins.pth')
    #print('Model parameters exported to dubins.pth')

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

plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Training Loss')
plt.title('Loss During Training', fontsize=16)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(valid_loss_history, label='Whole Data Loss')
plt.title('Loss During Testing', fontsize=16)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
    

