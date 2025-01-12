def prune_dcp_resnet50(model, pruning_ratio=0.5):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            weight_sum = module.weight.abs().sum((1, 2, 3))
            _, indices = torch.topk(weight_sum, int(module.out_channels * (1 - pruning_ratio)))
            module.out_channels = len(indices)
            module.weight.data = module.weight.data[indices, :, :, :]
    return model

# Load and prune the model
resnet50_dcp = models.resnet50(pretrained=True)
pruned_resnet50_dcp = prune_dcp_resnet50(resnet50_dcp, 0.5)
torch.save(pruned_resnet50_dcp.state_dict(), 'resnet50_dcp.pth')
