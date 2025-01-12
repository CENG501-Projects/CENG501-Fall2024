def prune_dcp_vgg16(model, pruning_ratio=0.5):
    for module in model.features:
        if isinstance(module, torch.nn.Conv2d):
            weight_sum = module.weight.abs().sum((1, 2, 3))
            _, indices = torch.topk(weight_sum, int(module.out_channels * (1 - pruning_ratio)))
            module.out_channels = len(indices)
            module.weight.data = module.weight.data[indices, :, :, :]
    return model

vgg16_dcp = models.vgg16(pretrained=True)
pruned_vgg16_dcp = prune_dcp_vgg16(vgg16_dcp, 0.5)
