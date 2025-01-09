
import os
import argparse
import time
import math
import pickle
import copy
from functools import partial
import torch

from src.experiments.model_initialization import model_initialization
from src.SRHM.optim_loss import loss_func, regularize, opt_algo, measure_accuracy
from src.SRHM.utils import cpu_state_dict, args2train_test_sizes
from src.SRHM.observables import locality_measure, state2permutation_stability, state2clustering_error




from src.SRHM.reference_code.hierarchical import RandomHierarchyModel as RHM
from src.SRHM.srhm import SparseRandomHierarchyModel as SRHM

# Models
def experiment_run(idx, experiment_name, args):

    args.ptr, args.pte = args2train_test_sizes(args)

    output_idx = f"outputs/{experiment_name}_{idx}.pkl"

    with open(output_idx, "wb+") as handle:
        pickle.dump(args, handle)
    try:
        for data in run(args):
            with open(output_idx, "wb") as handle:
                pickle.dump(data, handle)
    except:
        os.remove(output_idx)
        raise

def run(args):

    # if args.dtype == 'float64':
    #     torch.set_default_dtype(torch.float64)
    # if args.dtype == 'float32':
    #     torch.set_default_dtype(torch.float32)
    # if args.dtype == 'float16':
    #     torch.set_default_dtype(torch.float16)

    best_acc = 0  # best test accuracy
    criterion = partial(loss_func, args)

    trainloader, testloader, net0 = experiment_initialization(args)

    # synonym_loader list

    if args.seed_diffeo:
        testdiffeo = diffeo_init(args)

    if args.seed_synonym:
        testsynonym = synonym_init(args)

    # scale batch size when larger than train-set size
    if (args.batch_size >= args.ptr) and args.scale_batch_size:
        args.batch_size = args.ptr // 2

    if args.save_dynamics:
        dynamics = [{"acc": 0.0, "epoch": 0., "net": cpu_state_dict(net0)}]
    else:
        dynamics = None

    loss = []
    terr = []
    locality = []
    stability = []
    clustering_error = []
    epochs_list = []

    best = dict()
    trloss_flag = 0

    for idx, (net, epoch, losstr, avg_epoch_time) in enumerate(train(args, trainloader, net0, criterion)):

        assert str(losstr) != "nan", "Loss is nan value!!"
        loss.append(losstr)
        epochs_list.append(epoch)

        # measuring locality for fcn nets
        if args.locality == 1:
            assert args.net == 'fcn', "Locality can only be computed for fcns !!"
            state = net.state_dict()
            hidden_layers = [state[k] for k in state if 'w' in k][:-2]
            with torch.no_grad():
                locality.append(locality_measure(hidden_layers, args)[0])

        # measure stability to semantically equivalent data realizations
        if args.stability == 1:
            state = net.state_dict()
            stability.append(state2permutation_stability(state, args))
        if args.clustering_error == 1:
            state = net.state_dict()
            clustering_error.append(state2clustering_error(state, args))

        if epoch % 10 != 0 and not args.save_dynamics: continue

        if testloader:
            acc = test(args, testloader, net, criterion, print_flag=epoch % 5 == 0)
        else:
            acc = torch.nan
        terr.append(100 - acc)

        if args.save_dynamics:
        #     and (
        #     epoch
        #     in (10 ** torch.linspace(-1, math.log10(args.epochs), 30)).int().unique()
        # ):
            # save dynamics at 30 log-spaced points in time
            dynamics.append(
                {"acc": acc, "epoch": epoch, "net": cpu_state_dict(net)}
            )
        if acc > best_acc:
            best["acc"] = acc
            best["epoch"] = epoch
            if args.save_best_net:
                best["net"] = cpu_state_dict(net)
            # if args.save_dynamics:
            #     dynamics.append(best)
            best_acc = acc
            print(f"BEST ACCURACY ({acc:.02f}) at epoch {epoch:.02f} !!", flush=True)

        out = {
            "args": args,
            "epoch": epochs_list,
            "train loss": loss,
            "terr": terr,
            "locality": locality,
            "stability": stability,
            "clustering_error": clustering_error,
            "dynamics": dynamics,
            "best": best,
        }

        if args.seed_diffeo:
            acc_diffeo = test(args, testdiffeo, net, criterion, print_flag=epoch % 5 == 0)
            sensitivity_diffeo = None
            out["acc_diffeo"] = acc_diffeo
            out["sensitivity_diffeo"] = sensitivity_diffeo

        if args.seed_synonym:
            acc_synonym = test(args, testsynonym, net, criterion, print_flag=epoch % 5 == 0)
            sensitivity_synonym = None
            out["acc_synonym"] = acc_synonym
            out["sensitivity_synonym"] = sensitivity_synonym

        yield out

        if args.test_acc_stop and acc >= args.test_acc_stop:
            break

        if args.p and idx*args.batch_size >= args.p:
            break

        if (losstr == 0 and args.loss == 'hinge') or (losstr < args.zero_loss_threshold and args.loss == 'cross_entropy'):
            trloss_flag += 1
            if trloss_flag >= args.zero_loss_epochs:
                break

    try:
        wo = weights_evolution(net0, net)
    except:
        print("Weights evolution failed!")
        wo = None

    if args.locality == 2:
        assert args.net == 'fcn', "Locality can only be computed for fcns !!"
        state = net.state_dict()
        hidden_layers = [state[k] for k in state if 'w' in k][:-2]
        with torch.no_grad():
            locality.append(locality_measure(hidden_layers, args)[0])

    if args.stability == 2:
        state = net.state_dict()
        stability.append(state2permutation_stability(state, args))

    if args.clustering_error == 2:
        state = net.state_dict()
        clustering_error.append(state2clustering_error(state, args))

    out = {
        "args": args,
        "epoch": epochs_list,
        "train loss": loss,
        "terr": terr,
        "locality": locality,
        "stability": stability,
        "clustering_error": clustering_error,
        "dynamics": dynamics,
        "init": cpu_state_dict(net0) if args.save_init_net else None,
        "best": best,
        "last": cpu_state_dict(net) if args.save_last_net else None,
        "weight_evo": wo,
        'avg_epoch_time': avg_epoch_time,
    }
    yield out


def train(args, trainloader, net0, criterion):

    net = copy.deepcopy(net0)

    optimizer, scheduler = opt_algo(args, net)
    print(f"Training for {args.epochs} epochs...")

    start_time = time.time()

    num_batches = math.ceil(args.ptr / args.batch_size)
    checkpoint_batches = torch.linspace(0, num_batches, 10, dtype=int)

    for epoch in range(args.epochs):

        # layerwise training
        if epoch % (args.epochs // args.num_layers + 1) == 0:
            if 'layerwise' in args.net:
                l = epoch // (args.epochs // args.num_layers + 1)
                net.init_layerwise_(l)

        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            train_loss += loss.detach().item()
            regularize(loss, net, args.weight_decay, reg_type=args.reg_type)
            loss.backward()
            optimizer.step()

            correct, total = measure_accuracy(args, outputs, targets, correct, total)

            # during first epoch, save some sgd steps instead of after whole epoch
            if epoch < 10 and batch_idx in checkpoint_batches and batch_idx != (num_batches - 1):
                yield net, epoch + (batch_idx + 1) / num_batches, train_loss / (batch_idx + 1), None

        avg_epoch_time = (time.time() - start_time) / (epoch + 1)

        if epoch % 5 == 0:
            print(
                f"[Train epoch {epoch+1} / {args.epochs}, {print_time(avg_epoch_time)}/epoch, ETA: {print_time(avg_epoch_time * (args.epochs - epoch - 1))}]"
                f"[tr.Loss: {train_loss * args.alpha / (batch_idx + 1):.03f}]"
                f"[tr.Acc: {100.*correct/total:.03f}, {correct} / {total}]",
                flush=True
            )

        scheduler.step()

        yield net, epoch + 1, train_loss / (batch_idx + 1), avg_epoch_time


def test(args, testloader, net, criterion, print_flag=True):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()

            correct, total = measure_accuracy(args, outputs, targets, correct, total)

        if print_flag:
            print(
                f"[TEST][te.Loss: {test_loss * args.alpha / (batch_idx + 1):.03f}]"
                f"[te.Acc: {100. * correct / total:.03f}, {correct} / {total}]",
                flush=True
            )

    return 100.0 * correct / total


# timing function
def print_time(elapsed_time):

    # if less than a second, print milliseconds
    if elapsed_time < 1:
        return f"{elapsed_time * 1000:.00f}ms"

    elapsed_seconds = round(elapsed_time)

    m, s = divmod(elapsed_seconds, 60)
    h, m = divmod(m, 60)

    elapsed_time = []
    if h > 0:
        elapsed_time.append(f"{h}h")
    if not (h == 0 and m == 0):
        elapsed_time.append(f"{m:02}m")
    elapsed_time.append(f"{s:02}s")

    return "".join(elapsed_time)


def weights_evolution(f0, f):
    s0 = f0.state_dict()
    s = f.state_dict()
    nd = 0
    for k in s:
        nd += (s0[k] - s[k]).norm() / s0[k].norm()
    nd /= len(s)
    return nd




def experiment_initialization(args):
    """
        Initialize dataset and architecture.
    """
    torch.manual_seed(args.seed_init)

    args.diffeo_init = False
    args.synonym_init = False

    trainset, testset, input_dim, ch = dataset_initialization(args)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    if testset:
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        testloader = None

    
    net = model_initialization(args, input_dim=input_dim, ch=ch)

    return trainloader, testloader, net

def diffeo_init(args):

    torch.manual_seed(args.seed_init)

    args.diffeo_init = True

    _, testset, _, _ = dataset_initialization(args)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    args.diffeo_init = False

    return testloader

def synonym_init(args):
    torch.manual_seed(args.seed_init)

    args.synonym_init = True

    _, testset, _, _ = dataset_initialization(args)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    args.synonym_init = True
    return testloader

# Adapted from : https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch
def get_truly_random_seed_through_os():
    """
    Usually the best random sample you could get in any programming language is generated through the operating system. 
    In Python, you can use the os module.

    source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    """
    RAND_SIZE = 4
    random_data = os.urandom(
        RAND_SIZE
    )  # Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed

def get_max_size(args):
    d = args.s**args.num_layers
    m_exp = (d-1)/(args.s - 1)
    return args.num_classes * args.m**m_exp

def dataset_initialization(args):
    """
    Initialize train and test loaders for chosen dataset and transforms.
    :param args: parser arguments (see main.py)
    :return: trainloader, testloader, image size, number of classes.
    """

    nc = args.num_classes

    seed_p = None
    if args.synonym_init:
        seed_p = args.seed_synonym
    elif args.diffeo_init:
        seed_p = args.seed_diffeo

    MAX_SIZE = get_max_size(args)

    transform = None
    if args.dataset == "rhm":
        DataModel = RHM
    elif args.dataset == "srhm":
        DataModel = SRHM


    if args.p and args.p > MAX_SIZE:
        concat_count = (args.p // MAX_SIZE) + 1
        joint_training_loader = DataModel(
                num_features=args.num_features,
                m=args.m,  # features multiplicity
                num_layers=args.num_layers,
                num_classes=nc,
                s=args.s, # tuples size
                s0=args.s0,
                input_format=args.input_format,
                whitening=args.whitening,
                seed=args.seed_init,
                train=True,
                transform=transform,
                testsize=args.pte,
                max_dataset_size=args.ptr + args.pte,
                seed_p=seed_p)
        for _ in concat_count:
            torch.random_seed(get_truly_random_seed_through_os())
            new_training_loader = DataModel(
                num_features=args.num_features,
                m=args.m,  # features multiplicity
                num_layers=args.num_layers,
                num_classes=nc,
                s=args.s, # tuples size
                s0=args.s0,
                input_format=args.input_format,
                whitening=args.whitening,
                seed=args.seed_init,
                train=True,
                transform=transform,
                testsize=1,
                max_dataset_size=1,
                seed_p=seed_p)

            joint_training_loader.append(new_training_loader)
            # TODO : Make joint training set

        trainset = joint_training_loader

        torch.manual_seed(args.seed_init)
        testset = DataModel(
                num_features=args.num_features,
                m=args.m,  # features multiplicity
                num_layers=args.num_layers,
                num_classes=nc,
                s=args.s, # tuples size
                s0=args.s0,
                input_format=args.input_format,
                whitening=args.whitening,
                seed=args.seed_init,
                train=False,
                transform=transform,
                testsize=args.pte,
                max_dataset_size=args.ptr+args.pte,
                seed_p=seed_p)

    else:
        trainset = DataModel(
            num_features=args.num_features,
            m=args.m,  # features multiplicity
            num_layers=args.num_layers,
            num_classes=nc,
            s=args.s, # tuples size
            s0=args.s0,
            input_format=args.input_format,
            whitening=args.whitening,
            seed=args.seed_init,
            train=True,
            transform=transform,
            testsize=args.pte,
            max_dataset_size=args.ptr+args.pte,
            seed_p=seed_p)
        
        if args.pte:
            testset = DataModel(
                num_features=args.num_features,
                m=args.m,  # features multiplicity
                num_layers=args.num_layers,
                num_classes=nc,
                s=args.s, # tuples size
                s0=args.s0,
                input_format=args.input_format,
                whitening=args.whitening,
                seed=args.seed_init,
                train=False,
                transform=transform,
                testsize=args.pte,
                max_dataset_size=args.ptr+args.pte
            )
        else:
            testset = None

    input_dim = trainset[0][0].shape[-1]
    ch = trainset[0][0].shape[-2] if args.input_format != 'long' else 0

    if args.loss == 'hinge':
        # change to binary labels
        trainset.targets = 2 * (torch.as_tensor(trainset.targets) >= nc // 2) - 1
        if testset:
            testset.targets = 2 * (torch.as_tensor(testset.targets) >= nc // 2) - 1

    P = len(trainset)

    # take random subset of training set
    torch.manual_seed(args.seed_trainset)
    perm = torch.randperm(P)
    if args.p and args.p > MAX_SIZE:
        trainset = torch.utils.data.Subset(trainset, perm)
    else:
        trainset = torch.utils.data.Subset(trainset, perm[:args.ptr])

    return trainset, testset, input_dim, ch


