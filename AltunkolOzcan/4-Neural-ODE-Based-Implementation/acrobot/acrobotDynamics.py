import torch
from acrobot.autoGenAcrobot import autoGen_acrobotDynamics
import numpy as np
def acrobotDynamics(z, u, p):
    """
    Computes the dynamics of the acrobot: double pendulum, two point masses, torque motor between links, no friction.

    INPUTS:
        z = [4,] = state vector (tensor)
        u = scalar = input torque (tensor)
        p = parameter dictionary:
            - 'm1': elbow mass
            - 'm2': wrist mass
            - 'g': gravitational acceleration
            - 'l1': length shoulder to elbow
            - 'l2': length elbow to wrist

    OUTPUTS:
        dz = [4,] = dz/dt = time derivative of the state (tensor)
    
    NOTES:
        States:
            1 = q1 = first link angle
            2 = q2 = second link angle
            3 = dq1 = first link angular rate
            4 = dq2 = second link angular rate

        Angles: measured from negative j axis with positive convention
    """

    # Ensure z is always a 2D tensor (batch dimension added for single input)
    if z.ndim == 1:  # Single input case, shape [4,]
        z = z.unsqueeze(0)  # Convert to shape [1, 4]

    # Now z is guaranteed to have shape [batch_size, 4]
    q1 = z[:, 0]  # First element across batch
    q2 = z[:, 1]  # Second element across batch
    dq1 = z[:, 2]  # Third element across batch
    dq2 = z[:, 3]  # Fourth element across batch


    # Compute dynamics matrices: Mass matrix (D), gravitational forces (G), and input (B)
    D, G, B = autoGen_acrobotDynamics(q1, q2, dq1, dq2, p['m1'], p['m2'], p['g'], p['l1'], p['l2'])
    # Solve for accelerations ddq
    #m,s,v = torch.linalg.svd(D)
    #rcondi = 1e-6
    B = B.unsqueeze(-1).expand(-1, -1, u.shape[0])
    u = u.view(1,1,-1)
    right_side = B * u - G.unsqueeze(1)
    #right_side = right_side.unsqueeze(1)
    #ddq, residual, rank, sing_val = torch.linalg.lstsq(D.permute(2,0,1) , right_side.permute(2,0,1), rcond = rcondi, driver='gelss')
    ddq = torch.linalg.solve(D.permute(2,0,1), right_side.permute(2,0,1))
    #dz = torch.cat((torch.tensor([dq1, dq2]), ddq))
    dq1_reshaped = dq1.view(z.shape[0], 1, 1)
    dq2_reshaped = dq2.view(z.shape[0], 1, 1)

    # Concatenate along the last dimension to form (1024, 1, 4)
    dz = torch.cat((dq1_reshaped, dq2_reshaped, ddq), dim=1)
    dz = dz.squeeze(2)
    #dz = dz.permute(0,2,1)
    return dz
#
#q1 = torch.tensor(-1.0, requires_grad=False)
#q2 = torch.tensor(-1.0, requires_grad=False)
#dq1 = torch.tensor(1.0, requires_grad=False)
#dq2 = torch.tensor(1.0, requires_grad=False)
#
#z = torch.stack([q1, q2, dq1, dq2])
##print(z)
#p = {
#    'm1': 10,
#    'm2': 20,
#    'g': 9,
#    'l1': 3,
#    'l2': 5
#}
#
#dydt = acrobotDynamics(z,5,p)
#print(dydt)