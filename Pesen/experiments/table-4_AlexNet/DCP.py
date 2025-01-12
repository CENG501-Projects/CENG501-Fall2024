def prune_dcp(model, pruning_ratio=0.5):
    for module in model.features:
        if isinstance(module, torch.nn.Conv2d):
            weight_sum = module.weight.abs().sum((1, 2, 3))
            _, indices = torch.topk(weight_sum, int(module.out_channels * (1 - pruning_ratio)))
            module.out_channels = len(indices)
            module.weight.data = module.weight.data[indices, :, :, :]
    return model

alexnet_dcp = models.alexnet(pretrained=True)
pruned_alexnet_dcp = prune_dcp(alexnet_dcp, 0.5)
