import torch
from .autoGenAcrobot import autoGen_acrobotDynamics

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

    # Extract states from z
    q1 = z[0]
    q2 = z[1]
    dq1 = z[2]
    dq2 = z[3]

    # Compute dynamics matrices: Mass matrix (D), gravitational forces (G), and input (B)
    D, G, B = autoGen_acrobotDynamics(q1, q2, dq1, dq2, p['m1'], p['m2'], p['g'], p['l1'], p['l2'])

    # Solve for accelerations ddq
    ddq = torch.linalg.solve(D, B * u - G)

    # Construct dz (the derivative of the state)
    dz = torch.cat((torch.tensor([dq1, dq2]), ddq))
    return dz
