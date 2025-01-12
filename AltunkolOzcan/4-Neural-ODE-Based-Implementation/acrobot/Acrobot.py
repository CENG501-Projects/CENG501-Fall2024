import torch
from torch.nn.functional import pad
from .acrobotDynamics import acrobotDynamics
from .acrobot_control import AcrobotController

# This file contains the necessary Prob structure for the OCFlow
class Acrobot:
    """
    attributes:
        xtarget  - target state
        d        - number of dimensions
        g        - gravity acceleration
        m1       - mass of the first link
        m2       - mass of the second link
        l1       - length of the first link
        l2       - length of the second link
        training - boolean, if set to True during training and False during validation

    methods:
        train()          - set to training mode
        eval()           - set to evaluation mode
        calcGradpH(x, p) - calculate gradient of Hamiltonian wrt p
        calcLHQW         - calculate the Lagrangian and Hamiltonian
        calcQ            - calculate the obstacle/terrain cost Q
        calcObstacle     - calculate the obstacle/terrain costs for single agent Q_i
        calcW            - calculate the interaction costs W
        calcCtrls        - calculate the controls
        ---------
        calcU            - calculate thrust u
        f                - helper function
    """

    def __init__(self, xtarget, utarget, R, P, u = 0, m1=1.0, m2 = 1.0, l1 =3, l2 = 5, g=9.81):
        self.xtarget = xtarget.squeeze() # G assumes the target is squeezed
        self.utarget = utarget.squeeze()
        self.d = xtarget.numel()
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.training = True # so obstacle settings and self.r vary during training and validation
        self.u = u
        self.R = R # weights for the running cost
        self.P = P # weights for the terminal cost
        self.props = {
            'm1': self.m1,
            'm2': self.m2,
            'g': self.g,
            'l1': self.l1,
            'l2': self.l2
        }
        self.controller_net = AcrobotController()

    def __repr__(self):
        return "Acrobot Object"

    def __str__(self):
        s = "Acrobot Object Optimal Control \n"
        return s

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def calcGradpH(self, x, p):
        # gradient of the Hamiltonian wrt p
        gradp = -acrobotDynamics(x,self.u, self.props)
        gradp = gradp.squeeze(1)
        return gradp
    
    def calcLHQW(self, x, p):
        u_difference = self.u  - self.utarget
        uTR = torch.matmul(u_difference.unsqueeze(1), self.R) 
        control_cost = torch.bmm(uTR.unsqueeze(1), u_difference.unsqueeze(2)).squeeze(2).squeeze(1)

        x_difference = torch.abs(self.xtarget - x)  # Batch-wise difference
        xTP = torch.matmul(x_difference.unsqueeze(1), self.P)  # (Batch, 1, Dim)
        state_cost = torch.bmm(xTP, x_difference.unsqueeze(2)).squeeze(2).squeeze(1)

        running_cost = control_cost

        p = p.unsqueeze(2)
        f = acrobotDynamics(x, self.u, self.props)
        f = f.unsqueeze(1)
        dot_product = torch.bmm(f, p).squeeze(2).squeeze(1)
        H = - running_cost - dot_product #Hamiltonian
        L = - H - dot_product
        return L,H

    def calcCtrls(self, x,p):
        # controller neural network 
        res = self.controller_net(x)
        self.u = res
        setattr(self, "u", res)
        return self.u