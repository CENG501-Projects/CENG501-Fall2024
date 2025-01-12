def prune_taylor_resnet50(model, pruning_ratio=0.5):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            gradients = torch.sum(module.weight.grad**2, dim=(1, 2, 3))
            sorted_indices = torch.argsort(gradients, descending=True)
            keep_indices = sorted_indices[:int(module.out_channels * (1 - pruning_ratio))]
            module.weight.data = module.weight.data[keep_indices, :, :, :]
            module.out_channels = len(keep_indices)
    return model

# Load and prune the model
resnet50_taylor = models.resnet50(pretrained=True)
pruned_resnet50_taylor = prune_taylor_resnet50(resnet50_taylor, 0.5)
torch.save(pruned_resnet50_taylor.state_dict(), 'resnet50_taylor.pth')
