from src.SRHM.cnn import CNN
from src.SRHM.fcn import FCN
from src.SRHM.lcn import LocallyHierarchicalNet as LCN
from src.models.efficientnet import EfficientNetB0
from src.models.vgg import VGG
from src.models.resnet import ResNet18, ResNet34

# 'LCN', 'CNN', 'VGG11', 'VGG16', 'ResNet18', 'ResNet34', 'EfficientNetB0'

import torch


def model_initialization(args, input_dim, ch):
    """
    Neural netowrk initialization.
    :param args: parser arguments
    :return: neural network as torch.nn.Module
    """

    num_outputs = 1 if args.loss == "hinge" else args.num_classes
    
    net_layers = args.num_layers
    filter_size = args.s
    if args.dataset == "srhm":
        filter_size = args.s*(args.s0 + 1)

    ### Define network architecture ###
    torch.manual_seed(args.seed_net)

    net = None

    if args.net == "fcn":
        net = FCN(
            num_layers=net_layers,
            input_channels=ch * input_dim,
            h=args.width,
            out_dim=num_outputs,
            bias=args.bias,
        )
    elif args.net == "lcn":
        net = LCN(
            num_layers=net_layers,
            input_channels=ch,
            h=args.width,
            filter_size=filter_size,
            out_dim=num_outputs,
            bias=args.bias,
        )
    elif args.net == "cnn":
        net = CNN(
            num_layers=net_layers,
            input_channels=ch,
            h=args.width,
            patch_size=filter_size,
            out_dim=num_outputs,
            bias=args.bias,
        )
    # Models 'VGG11', 'VGG16', 'ResNet18', 'ResNet34', 'EfficientNetB0'
    elif args.net == "VGG11":
        net = VGG(vgg_name="VGG11", num_classes=num_outputs, num_ch=args.width)
    elif args.net == "VGG16":
        net = VGG(vgg_name="VGG16", num_classes=num_outputs)
    elif args.net == "ResNet18":
        net = ResNet18(num_classes=num_outputs)
    elif args.net == "ResNet34":
        net = ResNet34(num_classes=num_outputs)
    elif args.net == "EfficientNetB0":
        net = EfficientNetB0(num_classes=num_outputs, num_ch=1)

    assert net is not None, "Network architecture not in the list!"

    if args.random_features:
        for param in [p for p in net.parameters()][:-2]:
            param.requires_grad = False

    net = net.to(args.device)

    # if args.device == "cuda":
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    return net