def prune_l1(model, pruning_ratio=0.5):
    for module in model.features:
        if isinstance(module, torch.nn.Conv2d):
            l1_norm = torch.sum(module.weight.abs(), dim=(1, 2, 3))
            sorted_indices = torch.argsort(l1_norm, descending=True)
            keep_indices = sorted_indices[:int(module.out_channels * (1 - pruning_ratio))]
            module.weight.data = module.weight.data[keep_indices, :, :, :]
            module.out_channels = len(keep_indices)
    return model

alexnet_l1 = models.alexnet(pretrained=True)
pruned_alexnet_l1 = prune_l1(alexnet_l1, 0.5)
