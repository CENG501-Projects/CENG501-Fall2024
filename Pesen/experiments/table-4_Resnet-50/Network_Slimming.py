def prune_network_slimming_resnet50(model, pruning_ratio=0.5):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            bn_layer = next((m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)), None)
            if bn_layer:
                gamma = bn_layer.weight.data.abs()
                sorted_indices = torch.argsort(gamma, descending=True)
                keep_indices = sorted_indices[:int(module.out_channels * (1 - pruning_ratio))]
                module.weight.data = module.weight.data[keep_indices, :, :, :]
                module.out_channels = len(keep_indices)
    return model

# Load and prune the model
resnet50_slimming = models.resnet50(pretrained=True)
pruned_resnet50_slimming = prune_network_slimming_resnet50(resnet50_slimming, 0.5)
torch.save(pruned_resnet50_slimming.state_dict(), 'resnet50_slimming.pth')
