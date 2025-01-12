def prune_network_slimming(model, pruning_ratio=0.5):
    for module in model.features:
        if isinstance(module, torch.nn.Conv2d):
            bn_layer = next((m for m in model.features if isinstance(m, torch.nn.BatchNorm2d)), None)
            if bn_layer:
                gamma = bn_layer.weight.data.abs()
                sorted_indices = torch.argsort(gamma, descending=True)
                keep_indices = sorted_indices[:int(module.out_channels * (1 - pruning_ratio))]
                module.weight.data = module.weight.data[keep_indices, :, :, :]
                module.out_channels = len(keep_indices)
    return model

alexnet_slimming = models.alexnet(pretrained=True)
pruned_alexnet_slimming = prune_network_slimming(alexnet_slimming, 0.5)
