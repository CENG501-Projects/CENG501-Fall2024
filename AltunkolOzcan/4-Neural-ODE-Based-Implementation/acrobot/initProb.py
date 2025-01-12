# initProb.py
# initialize the OC problem

import torch
from .Acrobot import Acrobot

def initProb(nTrain, nVal, var0, cvt):
    """
    initialize the OC problem that we want to solve
    :param nTrain: int, number of samples in a batch, drawn from rho_0
    :param nVal:   int, number of validation samples to draw from rho_0
    :param var0: float, variance of rho_0
    :param alph:  list, 6-value list of parameters/hyperparameters
    :param cvt:   func, conversion function for typing and device
    :return:
        prob:  the problem Object
        x0:    nTrain -by- d tensor, training batch
        x0v:   nVal -by- d tensor, training batch
        xInit: 1 -by- d tensor, center of rho_0
    """
    
    xtarget = cvt(torch.tensor([[torch.pi, torch.pi, 0, 0]]))
    utarget = cvt(torch.tensor([0]))
    d = 4
    xInit = cvt(torch.tensor([[0, 0,0, 0]]))
    x0      = xInit + cvt(var0 * torch.randn(nTrain, d))
    x0v     = xInit + cvt(var0 * torch.randn(nVal, d))
    R = torch.tensor([0.01], dtype = torch.double)
    P = torch.eye(4)
    prob = Acrobot(xtarget=xtarget, utarget=utarget, R = R, P = P)
    return prob, x0, x0v, xInit


def resample(x0, xInit, var0, cvt):
    """
    resample rho_0 for next training batch

    :param x0:    nTrain -by- d tensor, previous training batch
    :param xInit: 1 -by- d tensor, center of rho_0
    :param var0: float, variance of rho_0
    :param cvt:   func, conversion function for typing and device
    :return: nTrain -by- d tensor, training batch
    """
    return xInit + cvt( var0 * torch.randn(*x0.shape) )