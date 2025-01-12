# plotter.py
# for generating plots

import matplotlib
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except:
    matplotlib.use('Agg')  # for linux server with no tkinter
    import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'inferno'

# avoid Type 3 fonts
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import os
import torch
from torch.nn.functional import pad
from mpl_toolkits.mplot3d import Axes3D  # for 3D plots
from PIL import Image, ImageDraw # for video

from .OCFlow import OCflow

def plotAcrobot(x, net, prob, nt, sPath, sTitle="", approach="ocflow"):
    xtarget = prob.xtarget

    if approach == 'ocflow':
        J,c, traj, trajCtrl = OCflow(x, net, prob, tspan=[0.0, 5.0], nt=nt, stepper="rk4", alph=net.alph, intermediates=True)
        trajCtrl = trajCtrl[:,:,1:] # want last dimension to be nt
    elif approach == 'baseline':
        # overload inputs to treat x and net differently for baseline
        traj = x # expects a tensor of size (nex, d, nt+1)
        trajCtrl = net # expects a tensor (nex, a, nt) where a is the dimension of the controls
    else:
        print("approach=" , approach, " is not an acceptable parameter value for plotAcrobot")    

    q1bounds = [-torch.pi, torch.pi]
    q2bounds = [-torch.pi, torch.pi]
    dq1bounds = [-3*torch.pi, 3*torch.pi]
    dq2bounds = [-3*torch.pi, 3*torch.pi]

    # make grid of plots
    nCol = 2
    nRow = 2

    fig = plt.figure(figsize=plt.figaspect(1.0))
    fig.set_size_inches(16, 8)
    fig.suptitle(sTitle)
    timet = range(nt)
    # positional movement training
    ax = fig.add_subplot(nRow, nCol, 1)
    ax.set_title('Acrobot Path')

    ax.plot(timet, traj[:,0,1:].view(-1).cpu().numpy(),label=r"$\theta_1$", linestyle='-', marker='o')
    ax.plot(timet, traj[:,1,1:].view(-1).cpu().numpy(),label=r"$\theta_2$", linestyle='-', marker='o')
    # ax.legend()

    # plot controls
    ax = fig.add_subplot(nRow, nCol, 2)
    
    # not using the t=0 values
    ax.plot(timet, trajCtrl[0, 0, :].cpu().numpy(), 'o-', label='u')
    ax.legend()
    ax.set_xticks([0, nt / 2, nt])
    ax.set_xlabel('nt')
    ax.set_ylabel('control')

    # plot L at each time step
    ax = fig.add_subplot(nRow, nCol, 3)
    timet = range(nt)
    # not using the t=0 values
    trajL = torch.sum(prob.R * trajCtrl[0, :, :] ** 2, dim=0, keepdims=True)
    totL = torch.sum(trajL[0, :]) / nt
    ax.plot(timet, trajL[0, :].cpu().numpy(), 'o-', label='L')
    ax.legend()
    ax.set_xticks([0, nt / 2, nt])
    ax.set_xlabel('nt')
    ax.set_ylabel('L')
    ax.set_title('L(x,T)=' + str(totL.item()))

    # plot velocities
    ax = fig.add_subplot(nRow, nCol, 4)
    timet = range(nt+1)
    ax.plot(timet, traj[0, 2, :].cpu().numpy(), 'o-', label=r'$dq1$')
    ax.plot(timet, traj[0, 3, :].cpu().numpy(), 'o-', label=r'$dq2$')
    ax.legend(ncol=2)
    ax.set_xticks([0, nt / 2, nt])
    ax.set_xlabel('nt')
    ax.set_ylabel('velocities')
    
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

