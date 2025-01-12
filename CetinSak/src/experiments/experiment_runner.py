
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


from src.SRHM.diffeomorphism_utilities import *

from src.SRHM.reference_code.hierarchical import RandomHierarchyModel as RHM
from src.SRHM.srhm import SparseRandomHierarchyModel as SRHM

# Models
def experiment_run(idx, experiment_name, args):

    args.ptr, args.pte = args2train_test_sizes(args)

    output_idx = f"outputs/{experiment_name}_{idx}.pkl"

    print(f"Checking if {output_idx} exists.")
    flag = 0
    try:
        with open(output_idx, "r") as handle:
            flag = 1
    except:
        pass
    if flag == 1:
        if args.skip_existing:
            print("")
            return
        raise OSError(f"File {output_idx} already exists, do you want to continue training with it?")

    print(f"{output_idx} does not exist, starting training.")
    with open(output_idx, "wb+") as handle:
        pickle.dump(args, handle)
    try:
        for data in run(args):
            with open(output_idx, "wb") as handle:
                pickle.dump(data, handle)
    except:
        print("Do you want to delete current results?")
        confirmation = input("Type 'ok' to delete file : ")
        if confirmation == 'ok':
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
        dynamics = [{"acc": 0.0, "epoch": 0.}]
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
        # if args.locality == 1:
        #     assert args.net == 'fcn', "Locality can only be computed for fcns !!"
        #     state = net.state_dict()
        #     hidden_layers = [state[k] for k in state if 'w' in k][:-2]
        #     with torch.no_grad():
        #         locality.append(locality_measure(hidden_layers, args)[0])

        # measure stability to semantically equivalent data realizations
        # if args.stability == 1:
        #     state = net.state_dict()
        #     stability.append(state2permutation_stability(state, args))
        # if args.clustering_error == 1:
        #     state = net.state_dict()
        #     clustering_error.append(state2clustering_error(state, args))

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
                {"acc": acc, "epoch": epoch,}
            )
        if acc > best_acc:
            best["acc"] = acc
            best["epoch"] = epoch
            # if args.save_dynamics:
            #     dynamics.append(best)
            best_acc = acc
            print(f"BEST ACCURACY ({acc:.02f}) at epoch {epoch:.02f} !!", flush=True)

        out = {
            "args": args,
            "epoch": epochs_list,
            "train loss": loss,
            "terr": terr,
            "dynamics": dynamics,
            "best": best,
        }

        if args.seed_diffeo:
            out["acc_diffeo"],out["sensitivity_diffeo"] = test_with_sensitivity(args, testdiffeo, net, criterion, print_flag=epoch % 5 == 0)


        if args.seed_synonym:
            acc_synonym = test(args, testsynonym, net, criterion, print_flag=epoch % 5 == 0)
            sensitivity_synonym = None
            out["acc_synonym"] = acc_synonym
            out["sensitivity_synonym"] = sensitivity_synonym

        yield out

        if args.test_acc_stop and acc >= args.test_acc_stop*100:
            break

        if args.p and idx*args.batch_size >= args.p:
            break

        if (losstr == 0 and args.loss == 'hinge') or (losstr < args.zero_loss_threshold and args.loss == 'cross_entropy'):
            trloss_flag += 1
            if trloss_flag >= args.zero_loss_epochs:
                break

    # try:
    #     wo = weights_evolution(net0, net)
    # except:
    #     print("Weights evolution failed!")
    #     wo = None

    # if args.locality == 2:
    #     assert args.net == 'fcn', "Locality can only be computed for fcns !!"
    #     state = net.state_dict()
    #     hidden_layers = [state[k] for k in state if 'w' in k][:-2]
    #     with torch.no_grad():
    #         locality.append(locality_measure(hidden_layers, args)[0])

    # if args.stability == 2:
    #     state = net.state_dict()
    #     stability.append(state2permutation_stability(state, args))

    # if args.clustering_error == 2:
    #     state = net.state_dict()
    #     clustering_error.append(state2clustering_error(state, args))

    out = {
        "args": args,
        "epoch": epochs_list,
        "train loss": loss,
        "terr": terr,
        "dynamics": dynamics,
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

    if num_batches >= 1000:
        checkpoint_batches = torch.linspace(0, num_batches, num_batches//100, dtype=int)
    if num_batches >= 1e6:
        checkpoint_batches = torch.linspace(0, num_batches, num_batches//1000, dtype=int)
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
            if args.net in ['VGG11', 'VGG16', 'ResNet18', 'ResNet34', 'EfficientNetB0']:
                inputs = inputs.unsqueeze(1)
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

def test_with_sensitivity_x(args, testloader, net, criterion, print_flag=True, num_transforms=10,sensitivity_type="diffeo"):

    if sensitivity_type == "diffeo":
        transform_type = apply_srhm_diffeomorphism
    elif sensitivity_type == "synonym":
        transform_type = apply_srhm_synonym
    else:
        raise ValueError("Unknown sensitivity type")

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_outputs = []
    all_inputs = []
    
    with torch.no_grad():
        # First pass: collect original outputs
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if args.net in ['VGG11', 'VGG16', 'ResNet18', 'ResNet34', 'EfficientNetB0']:
                inputs = inputs.unsqueeze(1)
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            correct, total = measure_accuracy(args, outputs, targets, correct, total)
            
            all_outputs.append(outputs)
            all_inputs.append(inputs)
            
            if print_flag and batch_idx % 10 == 0:
                print(
                    f"[TEST][te.Loss: {test_loss * args.alpha / (batch_idx + 1):.03f}]"
                    f"[te.Acc: {100. * correct / total:.03f}]",
                    flush=True
                )
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_inputs = torch.cat(all_inputs, dim=0)
        
        all_transformed_outputs = []
        transformed_outputs = []
        for inputs in all_inputs.split(args.batch_size):
            inputs_transformed = apply_srhm_diffeomorphism(
                inputs, args.s, args.s0,
            )
            
            #Second pass: collect transformed outputs
            transformed_outputs.append(net(inputs_transformed))
        all_transformed_outputs.append(torch.cat(transformed_outputs, dim=0))
        
        diff_sum = sum(torch.norm(all_outputs - transformed, dim=1).mean() 
                      for transformed in all_transformed_outputs)
        numerator = diff_sum / num_transforms
        
        idx_shuffle = torch.randperm(all_outputs.size(0))
        denominator = torch.norm(all_outputs - all_outputs[idx_shuffle], dim=1).mean()
        sensitivity = numerator / denominator

## Will be deprecated use above function instead later.
def test_with_sensitivity(args, testloader, net, criterion, print_flag=True, num_transforms=10):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_outputs = []
    all_inputs = []
    
    with torch.no_grad():
        # First pass: collect original outputs
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if args.net in ['VGG11', 'VGG16', 'ResNet18', 'ResNet34', 'EfficientNetB0']:
                inputs = inputs.unsqueeze(1)
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            correct, total = measure_accuracy(args, outputs, targets, correct, total)
            
            all_outputs.append(outputs)
            all_inputs.append(inputs)
            
            if print_flag and batch_idx % 10 == 0:
                print(
                    f"[TEST][te.Loss: {test_loss * args.alpha / (batch_idx + 1):.03f}]"
                    f"[te.Acc: {100. * correct / total:.03f}]",
                    flush=True
                )
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_inputs = torch.cat(all_inputs, dim=0)
        
        all_transformed_outputs = []
        
        # for _ in range(num_transforms) ## More than one diffeo...
        transformed_outputs = []
        for inputs in all_inputs.split(args.batch_size):
            inputs_transformed = apply_srhm_diffeomorphism(
                inputs, args.s, args.s0,
            )
            
            #Second pass: collect transformed outputs
            transformed_outputs.append(net(inputs_transformed))
        all_transformed_outputs.append(torch.cat(transformed_outputs, dim=0))
        
        diff_sum = sum(torch.norm(all_outputs - transformed, dim=1).mean() 
                      for transformed in all_transformed_outputs)
        numerator = diff_sum / num_transforms
        
        idx_shuffle = torch.randperm(all_outputs.size(0))
        denominator = torch.norm(all_outputs - all_outputs[idx_shuffle], dim=1).mean()
        sensitivity = numerator / denominator
    
    print("Sensitivity: ", sensitivity.item())
    return 100.0 * correct / total, sensitivity.item()

def test(args, testloader, net, criterion, print_flag=True):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if args.net in ['VGG11', 'VGG16', 'ResNet18', 'ResNet34', 'EfficientNetB0']:
                inputs = inputs.unsqueeze(1)
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
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print("Lenght of trainloader:",len(trainloader))

    if testset:
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        print("Lenght of testloader:",len(testloader))
    else:
        testloader = None

    
    net = model_initialization(args, input_dim=input_dim, ch=ch)

    return trainloader, testloader, net

def diffeo_init(args):

    torch.manual_seed(args.seed_init)

    args.diffeo_init = True

    _, testset, _, _ = dataset_initialization(args)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    args.diffeo_init = False

    return testloader

def synonym_init(args):
    torch.manual_seed(args.seed_init)

    args.synonym_init = True

    _, testset, _, _ = dataset_initialization(args)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    args.synonym_init = True
    return testloader

def get_max_size(args):
    d = args.s**args.num_layers
    m_exp = (d-1)/(args.s - 1)
    return args.num_classes * args.m**m_exp

def map_for_nets(x, y):
    return x.unsqueeze(1), y

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
        print(f"P = {args.p} is larger than MAX_SIZE = {MAX_SIZE}, training will be multiple epochs : {args.p/MAX_SIZE:.2f}")

    trainset = None
    if not (args.diffeo_init or args.synonym_init):
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
            testsize=-1,
            max_dataset_size=args.ptr+args.pte)
    
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
            testsize=-1,
            max_dataset_size=args.ptr+args.pte,
            seed_p=seed_p
        )
    else:
        testset = None

    input_dim, ch = None, None
    if trainset:
        input_dim = trainset[0][0].shape[-1]
        ch = trainset[0][0].shape[-2] if args.input_format != 'long' else 0

        print(f"Input dim : {input_dim}")
        print(f"ch : {ch}")

        if args.loss == 'hinge':
            # change to binary labels
            trainset.targets = 2 * (torch.as_tensor(trainset.targets) >= nc // 2) - 1
            if testset:
                testset.targets = 2 * (torch.as_tensor(testset.targets) >= nc // 2) - 1

        P = len(trainset)

        # take random subset of training set
        torch.manual_seed(args.seed_trainset)
        perm = torch.randperm(P)
        trainset = torch.utils.data.Subset(trainset, perm[:args.ptr])

    return trainset, testset, input_dim, ch


