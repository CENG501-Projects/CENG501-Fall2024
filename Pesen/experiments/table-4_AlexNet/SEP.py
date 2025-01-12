def prune_sep(model, pruning_ratio=0.5):
    for module in model.features:
        if isinstance(module, torch.nn.Conv2d):
            weights = module.weight
            entropy = torch.sum(weights**2, dim=(1, 2, 3))
            sorted_indices = torch.argsort(entropy, descending=True)
            keep_indices = sorted_indices[:int(module.out_channels * (1 - pruning_ratio))]
            module.weight.data = module.weight.data[keep_indices, :, :, :]
            module.out_channels = len(keep_indices)
    return model

alexnet_sep = models.alexnet(pretrained=True)
pruned_alexnet_sep = prune_sep(alexnet_sep, 0.5)
