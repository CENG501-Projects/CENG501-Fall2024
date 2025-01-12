def prune_l1_vgg16(model, pruning_ratio=0.5):
    for module in model.features:
        if isinstance(module, torch.nn.Conv2d):
            l1_norm = torch.sum(module.weight.abs(), dim=(1, 2, 3))
            sorted_indices = torch.argsort(l1_norm, descending=True)
            keep_indices = sorted_indices[:int(module.out_channels * (1 - pruning_ratio))]
            module.weight.data = module.weight.data[keep_indices, :, :, :]
            module.out_channels = len(keep_indices)
    return model

vgg16_l1 = models.vgg16(pretrained=True)
pruned_vgg16_l1 = prune_l1_vgg16(vgg16_l1, 0.5)
