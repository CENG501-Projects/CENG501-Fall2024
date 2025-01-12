def prune_sep_resnet50(model, pruning_ratio=0.5):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            weights = module.weight
            entropy = torch.sum(weights**2, dim=(1, 2, 3))
            sorted_indices = torch.argsort(entropy, descending=True)
            keep_indices = sorted_indices[:int(module.out_channels * (1 - pruning_ratio))]
            module.weight.data = module.weight.data[keep_indices, :, :, :]
            module.out_channels = len(keep_indices)
    return model

# Load and prune the model
resnet50_sep = models.resnet50(pretrained=True)
pruned_resnet50_sep = prune_sep_resnet50(resnet50_sep, 0.5)
torch.save(pruned_resnet50_sep.state_dict(), 'resnet50_sep.pth')
