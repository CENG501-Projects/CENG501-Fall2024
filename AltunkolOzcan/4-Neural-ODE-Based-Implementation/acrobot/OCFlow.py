# OCflow.py
import math
import torch
from torch.nn.functional import pad
from .valueFunction import *

def OCflow(x, Phi, prob, tspan , nt, stepper="rk4", alph =[1.0,1.0,0.01,0.01], intermediates=True, noMean=False ):
    """
        main workhorse of the approach

    :param x:       input data tensor nex-by-d
    :param Phi:     neural network
    :param xtarget: target state for OC problem
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list, the alpha value multipliers
    :param intermediates: if True, return the states and controls instead
    :param noMean: if True, do not compute the mean across samples for Jc and cs
    :return:
        Jc - float, objective function value dot(alph,cs)
        cs - list of the computed costs
    """
    nex, d = x.shape
    h = (tspan[1]-tspan[0]) / nt

    z = pad(x, [0, 1, 0, 0], value=0)
    P = Phi.getGrad(z)  # initial condition gradPhi = p
    P = P[:, 0:d]

    # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
    # nex - by - (2*d + 4)
    z = torch.cat( (x , torch.zeros(nex,4, dtype=x.dtype,device=x.device)) , 1)

    tk = tspan[0]

    if intermediates: # save the intermediate values as well
        # make tensor of size z.shape[0], z.shape[1], nt
        zFull = torch.zeros( *z.shape , nt+1, device=x.device, dtype=x.dtype)
        zFull[:,:,0] = z
        # hold the controls/thrust and torques on the path
        tmp = x[:,1].unsqueeze(1)
        ctrlFull = torch.zeros(*tmp.shape, nt + 1, dtype=x.dtype,device=x.device)

    for k in range(nt):
        if stepper == 'rk4':
            cntrl = prob.calcCtrls(z[:,:d],0)
            z = stepRK4(ocOdefun, z, Phi, prob, alph, tk, tk + h)
        elif stepper == 'rk1':
            cntrl = prob.calcCtrls(z[k,:d],0)
            z = stepRK1(ocOdefun, z, Phi, prob, alph, tk, tk + h)
        tk += h
        if intermediates:
            zFull[:, :, k+1] = z
            tmp = pad(z[:,0:d], [0, 1, 0, 0], value=tk-h)
            p = Phi.getGrad(tmp)[:,0:d]
            ctrlFull[:, :, k + 1] = prob.u


    resG = ocG(z[:,0:d], prob.xtarget)
    cG   = torch.sum(resG**2, 1, keepdims=True)

    # compute Phi at final time
    tmp = pad(z[:,0:d], [0, 1, 0, 0], value=tspan[1])
    Phi1 = Phi(tmp)
    gradPhi1 = Phi.getGrad(tmp)[:, 0:d]

    if noMean:
        costAngle = getAngularDiff(z[:,0:d],prob.xtarget).view(-1,1)
        costL = z[:, d].view(-1,1)
        costG = cG.view(-1,1)
        costHJt = z[:, d+1].view(-1,1)
        costHJf = torch.sum(torch.abs(Phi1 - cG), 1).view(-1,1)
        costHJgrad = torch.sum(torch.abs(gradPhi1 - resG), 1).view(-1,1)
        cs = [costL, costG, costHJt, costHJf, costHJgrad]
        Jc = alph[0]*(costL + costG + costAngle) + alph[1]*costHJt + alph[2]*costHJf + alph[3]*costHJgrad
        return Jc, cs


    # ASSUME all examples are equally weighted
    costAngle = getAngularDiff(z[:,0:d],prob.xtarget)
    costAngle = torch.mean(costAngle)
    costL   = torch.mean(z[:,d])
    costG   = torch.mean(cG)
    costHJt = torch.mean(z[:,d+1])
    costHJf = torch.mean(torch.sum(torch.abs(Phi1 -  cG), 1))
    costHJgrad = torch.mean(torch.sum(torch.abs(gradPhi1 - resG), 1))

    cs = [costL, costG, costHJt, costHJf, costHJgrad]
    # Jc = sum(i[0] * i[1] for i in zip(cs, alph))
    Jc = alph[0]*(costL + costG + costAngle) + alph[1]*costHJt + alph[2]*costHJf + alph[3]*costHJgrad

    if intermediates:
        return Jc, cs, zFull, ctrlFull
    else:
        return Jc, cs

def ocG(z, xtarget):
    """G for OC problems"""

    d = xtarget.shape[0] # assumes xtarget has only one dimension

    vel_diff = z[:,2:d] - xtarget[2:d]
    angle_diff = torch.abs(z[:,0:2] - xtarget[0:2])
    realp = torch.cos(angle_diff)
    imagp = torch.sin(angle_diff)
    comp = torch.complex(realp, imagp)
    angle_res = torch.angle(comp)
    res = z
    res[:,0:2] = angle_res
    res[:,2:d] =vel_diff
    return res


def ocOdefun(x, t, net, prob, alph=None):
    """
    the diffeq function for the 4 ODEs in one

    d_t  [z_x ; L_x ; hjt_x ; dQ_x ; dW_x] = odefun( [z_x ; L_x ; hjt_x ; dQ_x ; dW_x] , t )

    z_x - state
    L_x - accumulated transport costs
    hjt_x - accumulated error between grad_t Phi and H

    :param x:    nex -by- d+4 tensor, state of diffeq
    :param t:    float, time
    :param net:  neural network Phi
    :param prob: problem Object
    :param alph: list, the 6 alpha values for the OC problem
    :return:
    """
    nex, d_extra = x.shape
    d = (d_extra - 4)

    z = pad(x[:, :d], (0, 1, 0, 0), value=t)  # concatenate with the time t

    gradPhi = net.getGrad(z)
    
    L, H = prob.calcLHQW(z[:,:d], gradPhi[:,0:d])

    res = torch.zeros(nex,d+4, dtype=x.dtype, device=x.device) # [dx ; dv ; hjt]

    res[:, 0:d]   = - prob.calcGradpH(z[:,:d] , gradPhi[:,0:d]) # dx
    res[:, d]     = L.squeeze()                                 # dv
    res[:, d + 1] = torch.abs( gradPhi[:,-1] - H.squeeze() )    # HJt

    return res   # torch.cat((dx, dv, hjt, f), 1)


def stepRK1(odefun, z, Phi, prob, alph, t0, t1):
    """
        Runge-Kutta 1 / Forward Euler integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 6 alpha values for the OC problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """
    z += (t1 - t0) * odefun(z, t0, Phi, prob, alph=alph)
    return z

def stepRK4(odefun, z, Phi, prob, alph, t0, t1):
    """
        Runge-Kutta 4 integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 6 alpha values for the OC problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """

    h = t1 - t0 # step size
    z0 = z

    K = h * odefun(z0, t0, Phi, prob, alph=alph)
    z = z0 + (1.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, prob, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, prob, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + K , t0+h , Phi, prob, alph=alph)
    z += (1.0/6.0) * K

    return z

def getAngularDiff(z,xtarget):
    d = xtarget.shape[0]
    angle_diff = torch.abs(z[:,0:2] - xtarget[0:2])
    realp = torch.cos(angle_diff)
    imagp = torch.sin(angle_diff)
    comp = torch.complex(realp, imagp)
    angle_res = torch.angle(comp)
    res = abs(angle_res[:,0]) + abs(angle_res[:,1])
    return res