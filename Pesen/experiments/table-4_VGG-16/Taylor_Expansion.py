def prune_taylor_vgg16(model, pruning_ratio=0.5):
    for module in model.features:
        if isinstance(module, torch.nn.Conv2d):
            gradients = torch.sum(module.weight.grad**2, dim=(1, 2, 3))
            sorted_indices = torch.argsort(gradients, descending=True)
            keep_indices = sorted_indices[:int(module.out_channels * (1 - pruning_ratio))]
            module.weight.data = module.weight.data[keep_indices, :, :, :]
            module.out_channels = len(keep_indices)
    return model

vgg16_taylor = models.vgg16(pretrained=True)
pruned_vgg16_taylor = prune_taylor_vgg16(vgg16_taylor, 0.5)
