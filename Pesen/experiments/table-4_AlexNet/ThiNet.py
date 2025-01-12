import torch
import torchvision.models as models
import numpy as np

def prune_thinet(model, pruning_ratio=0.5):
    for module in model.features:
        if isinstance(module, torch.nn.Conv2d):
            num_channels = module.out_channels
            num_to_prune = int(num_channels * pruning_ratio)
            module.out_channels -= num_to_prune
            module.weight.data = module.weight.data[:module.out_channels, :, :, :]
    return model

alexnet_thinet = models.alexnet(pretrained=True)
pruned_alexnet_thinet = prune_thinet(alexnet_thinet, 0.5)
