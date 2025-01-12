import torch
import torchvision.models as models

def prune_thinet_resnet50(model, pruning_ratio=0.5):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            num_channels = module.out_channels
            num_to_prune = int(num_channels * pruning_ratio)
            module.out_channels -= num_to_prune
            module.weight.data = module.weight.data[:module.out_channels, :, :, :]
    return model

# Load and prune the model
resnet50_thinet = models.resnet50(pretrained=True)
pruned_resnet50_thinet = prune_thinet_resnet50(resnet50_thinet, 0.5)
torch.save(pruned_resnet50_thinet.state_dict(), 'resnet50_thinet.pth')
