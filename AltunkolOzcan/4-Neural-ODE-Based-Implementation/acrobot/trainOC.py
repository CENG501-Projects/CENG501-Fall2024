# trainOC.py
# train neural network driver for optimal control problems

import argparse
import numpy as np
import time
import datetime
import torch
import os
import acrobot.utils as utils
from acrobot.utils import count_parameters
from torch.nn.functional import pad

from acrobot.valueFunction import *
from acrobot.OCFlow import OCflow
from acrobot.plotter import *
from acrobot.initProb import *


# defaults are for

parser = argparse.ArgumentParser('Optimal Control')
parser.add_argument(
    '--data',type=str, default='acrobot')

parser.add_argument("--nt"    , type=int, default=250, help="number of time steps")
parser.add_argument("--nt_val", type=int, default=250, help="number of time steps for validation")
parser.add_argument('--alph'  , type=str, default='1.0, 10.0, 0.01, 0.01') 
# alph order: G, HJt, HJfin, HJgrad
parser.add_argument('--m'     , type=int, default=20, help="NN width")
parser.add_argument('--nTh'   , type=int, default=10 , help="NN depth")

parser.add_argument('--niters', type=int, default=5000)
parser.add_argument('--lr'    , type=float, default=0.01)
parser.add_argument('--optim' , type=str, default='adam', choices=['adam'])
parser.add_argument('--weight_decay', type=float, default=0.0)

parser.add_argument('--resume'  , type=str, default="experiments/oc/runU250V7/acrobot_2025_01_12_16_27_22_alph2_0_0_0_m20_checkpt.pth", help="for loading a pretrained model")
parser.add_argument('--save'    , type=str, default='experiments/oc/runU250V8', help="define the save directory")
parser.add_argument('--gpu'     , type=int, default=0, help="send to specific gpu")
parser.add_argument('--prec'    , type=str, default='double', choices=['single','double'], help="single or double precision")
parser.add_argument('--approach', type=str, default='ocflow', choices=['ocflow'])

parser.add_argument('--viz_freq', type=int, default=100, help="how often to plot visuals") # must be >= val_freq
parser.add_argument('--val_freq', type=int, default=25, help="how often to run model on validation set")
parser.add_argument('--log_freq', type=int, default=10, help="how often to print results to log")

parser.add_argument('--lr_freq' , type=int  , default=300, help="how often to decrease lr")
parser.add_argument('--lr_decay', type=float, default=0.95, help="how much to decrease lr")
parser.add_argument('--n_train' , type=int  , default=32, help="number of training samples")
parser.add_argument('--var0'    , type=float, default=1.0, help="variance of rho_0 to sample from")
parser.add_argument('--sample_freq',type=int, default=150, help="how often to resample training data")

# to adjust alph weights midway through training, which can be helpful in more complicated problems
# where the first task is to get to a low G, then tune things to be more optimal
# the args.resume flag allows for more functionality, and this is included so training can be one linux command
parser.add_argument('--new_alph', type=str, default=None, help='switch alph weights during training')
# example: default='12, 100.0, 10000.0, 300.0, 0.0')


args = parser.parse_args()

args.alph = [float(item) for item in args.alph.split(',')]

if args.new_alph is not None:
    args.new_alph = [float(item) for item in args.new_alph.split(',')]
    nChgAlpha = int(args.new_alph[0])
else:
    nChgAlpha = -1

if args.prec =='double':
    argPrec = torch.float64
else:
    argPrec = torch.float32


# add timestamp to save path
sStartTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info("start time: " + sStartTime)
logger.info(args)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
print('device: ',device)

if __name__ == '__main__':

    torch.set_default_dtype(argPrec)
    cvt = lambda x: x.type(argPrec).to(device, non_blocking=True)

    n_train = args.n_train
    nVal    = n_train
    alph    = args.alph

    # set-up problem nTrain, nVal, var0, cvt
    prob, x0, x0v, xInit = initProb( n_train, nVal, var0=args.var0, cvt=cvt)

    # set-up model
    d      = x0.size(1) # dimension of the problem
    m      = args.m
    nt     = args.nt
    nt_val = args.nt_val
    nTh    = args.nTh
    tspan  = [0.0, 5.0] # throughout we solve on [ 0 , T=1 ]

    net = Phi(nTh=nTh,d=d,m=m, alph=alph)
    net = net.to(argPrec).to(device)

    # resume training on a model that's already had some training
    if args.resume is not None:
        logger.info(' ')
        logger.info("loading model: {:}".format(args.resume))
        logger.info(' ')

        # load model
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        m       = checkpt['args'].m
        alph    = args.alph # overwrite saved alpha
        # alph  = checkpt['args'].alph # load alphas from resume model
        nTh     =  checkpt['args'].nTh
        net     = Phi(nTh=nTh, m=m, d=d, alph=alph)  # the Phi aka the value function
        net.load_state_dict(checkpt["state_dict"])
        net = net.to(argPrec).to(device)

    optim = torch.optim.Adam(list(net.parameters()) + list(prob.controller_net.parameters()), lr=args.lr, weight_decay=args.weight_decay )

    strTitle = args.data + '_' + sStartTime + '_alph{:}_{:}_{:}_{:}_m{:}'.format(
                     int(alph[0]), int(alph[1]), int(alph[2]),int(alph[3]), m)

    logger.info(net)
    logger.info("--------------------------------------------------")
    logger.info(prob)
    logger.info("--------------------------------------------------")
    logger.info("DIMENSION={:}  m={:}  nTh={:}   alpha={:}".format(d,m,nTh,alph))
    logger.info("nt={:}   nt_val={:}".format(nt,nt_val))
    logger.info("Number of trainable parameters: {}".format(count_parameters(net)))
    logger.info("--------------------------------------------------")
    logger.info(str(optim)) # optimizer info
    logger.info("data={:} device={:}".format(args.data, device))
    logger.info("n_train={:}".format(n_train))
    logger.info("maxIters={:} val_freq={:} viz_freq={:}".format(args.niters, args.val_freq, args.viz_freq))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info(strTitle)
    logger.info("--------------------------------------------------\n")

    # show Q and W values, but they're already included inside the L value
    log_msg = (
            '{:5s} {:7s} {:6s}   {:9s}  {:8s}  {:8s}  {:8s}  {:8s}  {:8s}  {:9s} {:8s}  {:8s}  {:8s}  {:8s}  {:8s} '.format(
            'iter', 'lr', '  time', 'loss', 'L', 'G', 'HJt', 'HJfin', 'HJgrad', 'valLoss', 'valL', 'valG', 'valHJt', 'valHJf','valHJg',
        )
    )

    logger.info(log_msg)

    best_loss = float('inf')
    bestParams = None
    time_meter = utils.AverageMeter()
    end = time.time()

    net.train()

    train_Jc_log, train_cs_log = [], [[] for _ in range(5)]
    test_Jc_log, test_cs_log = [], [[] for _ in range(5)]
    train_iters, test_iters = [], []

    for itr in range(1, args.niters+1):

        optim.zero_grad()
        Jc, cs, zFull, ctrlFull = OCflow(x0, net, prob, tspan=tspan, nt=nt, stepper="rk4", alph=net.alph)
        Jc.backward()
        optim.step()  # ADAM

        train_Jc_log.append(Jc.item())
        for i in range(5):
            train_cs_log[i].append(cs[i].item())
        train_iters.append(itr)

        time_meter.update(time.time() - end)

        log_message = (
            '{:05d} {:7.1e} {:6.2f}   {:9.3e}  {:8.2e}  {:8.2e}  {:8.2e}  {:8.2e}  {:8.2e}'.format(
                itr, optim.param_groups[0]['lr'], time_meter.val, Jc, cs[0], cs[1], cs[2], cs[3], cs[4]
            )
        )


        # validation
        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                net.eval()
                prob.eval()

                test_loss, test_cs, test_zFull, test_ctrlFull = OCflow(x0v, net, prob, tspan=tspan, nt=nt, stepper="rk4", alph=net.alph)
                # Log testing data
                test_Jc_log.append(test_loss.item())
                for i in range(5):
                    test_cs_log[i].append(test_cs[i].item())
                test_iters.append(itr)
                # add to print message
                log_message += '    {:9.2e}  {:8.2e}  {:8.2e}  {:8.2e}  {:8.2e}  {:8.2e}'.format(
                    test_loss, test_cs[0], test_cs[1], test_cs[2], test_cs[3], test_cs[4]
                )

                # save best set of parameters
                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    best_cs = test_cs
                    utils.makedirs(args.save)
                    bestParams = net.state_dict()
                    torch.save({
                        'args': args,
                        'state_dict': bestParams,
                        }, os.path.join(args.save, strTitle + '_checkpt.pth'))

                net.train()
                prob.train()


        if itr % args.log_freq == 0: # wait several iteration to print
            logger.info(log_message)  # print iteration

        # make a plot
        if itr % args.viz_freq == 0:
            # Plot for Jc
            JCfigpath = args.save + '/figs/' + "Jc_iter_" +  str(itr) +  '.png'
            if not os.path.exists(os.path.dirname(JCfigpath)):
                os.makedirs(os.path.dirname(JCfigpath))
            plt.figure(figsize=(8, 5))
            plt.plot(train_iters, train_Jc_log, label="Train Jc", linestyle='-', marker='.')
            if test_Jc_log:
                plt.plot(test_iters, test_Jc_log, label="Test Jc", linestyle='--', marker='o')
            plt.xscale("linear")
            plt.yscale("log")
            plt.xlabel("Iteration")
            plt.ylabel("Jc (log scale)")
            plt.title("Jc over Iterations")
            plt.legend()
            plt.grid(True)
            plt.savefig(JCfigpath, dpi = 300)
            plt.close()

            # Plot for cs[0] to cs[4]
            for i in range(5):
                csfigpath = args.save + '/figs/' + "cs" +  str(i) +"_iter"+str(itr)+  '.png'
                if not os.path.exists(os.path.dirname(csfigpath)):
                    os.makedirs(os.path.dirname(csfigpath))
                plt.figure(figsize=(8, 5))
                plt.plot(train_iters, train_cs_log[i], label=f"Train cs[{i}]", linestyle='-', marker='.')
                if test_cs_log[i]:
                    plt.plot(test_iters, test_cs_log[i], label=f"Test cs[{i}]", linestyle='--', marker='o')
                plt.xscale("linear")
                plt.yscale("log")
                plt.xlabel("Iteration")
                plt.ylabel(f"cs[{i}] (log scale)")
                plt.title(f"cs[{i}] over Iterations")
                plt.legend()
                plt.grid(True)
                plt.savefig(csfigpath, dpi = 300)
                plt.close()

            ##
            net.eval()
            prob.eval()
            
            currState = net.state_dict()
            net.load_state_dict(bestParams)

            with torch.no_grad():

                nSamples = 1
                # include parameters in file name
                sPath = args.save + '/figs/' + strTitle + "-iter" + str(itr) +  '.png'
                x0v2 = cvt(torch.cat((args.var0 * torch.randn(nSamples, 4), torch.zeros(nSamples, d - 4)), dim=1))
                x0v2 = xInit + x0v2
                x0v2[0,:] = xInit
                plotAcrobot(x0v2, net, prob, nt_val, sPath, sTitle=strTitle, approach=args.approach)
                
            net.load_state_dict(currState)
            net.train()
            prob.train()

        # shrink step size
        if itr % args.lr_freq == 0:
            net.load_state_dict(bestParams) # reset parameters to the best so far
            for p in optim.param_groups:
                p['lr'] *= args.lr_decay

        if itr % args.sample_freq == 0:
            x0 = resample(x0, xInit, args.var0, cvt)

        # change weights on penalizers midway through training
        if itr == nChgAlpha:
            alph = args.new_alph[1:]
            prob, _, _, _ = initProb( n_train, nVal, var0=args.var0, cvt=cvt)

            logger.info('alph values changed')

        end = time.time()

    logger.info("Training Time: {:} seconds".format(time_meter.sum))
    logger.info('Training has finished.  ' + os.path.join(args.save, strTitle ))

