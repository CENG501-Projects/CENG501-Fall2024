import torch
import torchvision.models as models

def prune_thinet_vgg16(model, pruning_ratio=0.5):
    for module in model.features:
        if isinstance(module, torch.nn.Conv2d):
            num_channels = module.out_channels
            num_to_prune = int(num_channels * pruning_ratio)
            module.out_channels -= num_to_prune
            module.weight.data = module.weight.data[:module.out_channels, :, :, :]
    return model

vgg16_thinet = models.vgg16(pretrained=True)
pruned_vgg16_thinet = prune_thinet_vgg16(vgg16_thinet, 0.5)
