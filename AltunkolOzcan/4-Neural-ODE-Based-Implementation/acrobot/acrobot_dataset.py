import numpy as np
import torch
from acrobotDynamicsLearn import acrobotDynamics

## Physical parameters
#p = {
#    'm1': 10,
#    'm2': 20,
#    'g': 9,
#    'l1': 3,
#    'l2': 5
#}
#
## Initial state
#q1 = (np.pi / 180) * 120
#q2 = (np.pi / 180) * 50
#dq1 = 0
#dq2 = 0
#z0 = np.array([q1, q2, dq1, dq2])  # Pack up initial state
#
#tSpan = [0, 128]  # Time span for the simulation ## UNUSED ANYWAY
#u = 0  # Motor torque (set to zero for passive dynamics)
#
## Define bounds and initialize dataset
#u_max = 5
#u_min = -5
#q_min = -np.pi
#q_max = np.pi
#dq_min = -10
#dq_max = 10
#dataset_length = 5000
#dataset = np.zeros((dataset_length, 10))
#
## Generate the dataset
#for i in range(dataset_length):
#    u = u_min + (u_max - u_min) * np.random.rand()  # Random motor torque
#
#    # Random initial state within specified bounds
#    x = np.array([q_min, q_min, dq_min, dq_min]) + \
#        (np.array([q_max, q_max, dq_max, dq_max]) - np.array([q_min, q_min, dq_min, dq_min])) * np.random.rand(4)
#
#    # Convert x and u to PyTorch tensors
#    z = torch.tensor(x, dtype=torch.float32, requires_grad=True)
#    u_tensor = torch.tensor(u, dtype=torch.float32, requires_grad=True)
#
#    # Get the dynamics output
#    f = acrobotDynamics(z, u_tensor, p)
#
#    # Store the data in the dataset
#    dataset[i, 0] = u
#    dataset[i, 1:5] = x
#    dataset[i, 5:9] = f.detach().numpy()
#
## Now `dataset` contains the generated data

def sampleYdotAcrobot(u,p, data_size):

    q_min = -np.pi
    q_max = np.pi
    dq_min = -10
    dq_max = 10
    ydot = torch.zeros((data_size,4,))
    y = torch.zeros((data_size,4,))
    for i in range(data_size):
        inp = u[i]
        #x = np.array([q_min, q_min, dq_min, dq_min]) + \
        #    (np.array([q_max, q_max, dq_max, dq_max]) - np.array([q_min, q_min, dq_min, dq_min])) * np.random.rand(4)
        q_range = torch.tensor([q_max - q_min, q_max - q_min, dq_max - dq_min, dq_max - dq_min], dtype=torch.float32)
        q_min_tensor = torch.tensor([q_min, q_min, dq_min, dq_min], dtype=torch.float32)
        x = q_min_tensor + q_range * torch.rand(4, dtype=torch.float32)
        y[i] = x
        # print(x)
        # Convert x and u to PyTorch tensors
        #z = x.clone().detach().requires_grad_(True)
        u_tensor = inp.clone().detach().requires_grad_(True)
        # Get the dynamics output
        f = acrobotDynamics(x, u_tensor, p) 
        ydot[i] = f       
    return y,ydot

