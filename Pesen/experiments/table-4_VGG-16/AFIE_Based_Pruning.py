from afie_and_pruning_ratio import LayerPruningAnalyzer

def prune_afie_vgg16(model, pruning_ratio=0.5):
    for module in model.features:
        if isinstance(module, torch.nn.Conv2d):
            weight_matrix = module.weight.view(module.weight.size(0), -1)
            analyzer = LayerPruningAnalyzer(weight_matrix)
            afie, _ = analyzer.compute_afie_and_pruning_ratio(pruning_ratio)
            sorted_indices = torch.argsort(weight_matrix.sum(1), descending=True)
            keep_indices = sorted_indices[:int(module.out_channels * (1 - pruning_ratio))]
            module.weight.data = module.weight.data[keep_indices, :, :, :]
            module.out_channels = len(keep_indices)
    return model

vgg16_afie = models.vgg16(pretrained=True)
pruned_vgg16_afie = prune_afie_vgg16(vgg16_afie, 0.5)
