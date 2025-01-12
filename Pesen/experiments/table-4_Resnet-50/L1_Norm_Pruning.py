def prune_l1_resnet50(model, pruning_ratio=0.5):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            l1_norm = torch.sum(module.weight.abs(), dim=(1, 2, 3))
            sorted_indices = torch.argsort(l1_norm, descending=True)
            keep_indices = sorted_indices[:int(module.out_channels * (1 - pruning_ratio))]
            module.weight.data = module.weight.data[keep_indices, :, :, :]
            module.out_channels = len(keep_indices)
    return model

# Load and prune the model
resnet50_l1 = models.resnet50(pretrained=True)
pruned_resnet50_l1 = prune_l1_resnet50(resnet50_l1, 0.5)
torch.save(pruned_resnet50_l1.state_dict(), 'resnet50_l1.pth')
