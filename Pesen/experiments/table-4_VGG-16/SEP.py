def prune_sep_vgg16(model, pruning_ratio=0.5):
    for module in model.features:
        if isinstance(module, torch.nn.Conv2d):
            weights = module.weight
            entropy = torch.sum(weights**2, dim=(1, 2, 3))
            sorted_indices = torch.argsort(entropy, descending=True)
            keep_indices = sorted_indices[:int(module.out_channels * (1 - pruning_ratio))]
            module.weight.data = module.weight.data[keep_indices, :, :, :]
            module.out_channels = len(keep_indices)
    return model

vgg16_sep = models.vgg16(pretrained=True)
pruned_vgg16_sep = prune_sep_vgg16(vgg16_sep, 0.5)
