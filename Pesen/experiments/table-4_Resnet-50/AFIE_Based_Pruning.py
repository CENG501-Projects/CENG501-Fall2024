from afie_and_pruning_ratio import LayerPruningAnalyzer

def prune_afie_resnet50(model, pruning_ratio=0.5):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            weight_matrix = module.weight.view(module.weight.size(0), -1)
            analyzer = LayerPruningAnalyzer(weight_matrix)
            afie, _ = analyzer.compute_afie_and_pruning_ratio(pruning_ratio)
            sorted_indices = torch.argsort(weight_matrix.sum(1), descending=True)
            keep_indices = sorted_indices[:int(module.out_channels * (1 - pruning_ratio))]
            module.weight.data = module.weight.data[keep_indices, :, :, :]
            module.out_channels = len(keep_indices)
    return model

# Load and prune the model
resnet50_afie = models.resnet50(pretrained=True)
pruned_resnet50_afie = prune_afie_resnet50(resnet50_afie, 0.5)
torch.save(pruned_resnet50_afie.state_dict(), 'resnet50_afie.pth')
