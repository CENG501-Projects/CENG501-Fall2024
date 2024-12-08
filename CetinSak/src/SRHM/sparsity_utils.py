import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from itertools import *
import random
import numpy as np
from src.utils.utils import number2base

def sample_hierarchical_rules_type_a(num_features, num_layers, m, num_classes, s, s0, seed=0):
    random.seed(seed)
    all_levels_paths = [torch.arange(num_classes)]
    all_levels_tuples = []
    
    for l in range(num_layers):
        old_paths = all_levels_paths[-1].flatten()
        old_features = list(set([i.item() for i in old_paths]))
        num_old_features = len(old_features)

        # Generate tuples with sparsity s(s_0 + 1) s informative s*s_0 uninformative
        sparse_tuple_size = s * (s0 + 1)
        possible_tuples = list(product(range(num_features), repeat=s))
        num_new_tuples = m * num_old_features

        assert len(possible_tuples) >= num_new_tuples

        random.shuffle(possible_tuples)
        selected_tuples = possible_tuples[:num_new_tuples]

        # uninformative features are randomly placed in the patch here.
        new_tuples = []
        for tup in selected_tuples:
            sparse_tup = [-1] * sparse_tuple_size  
            for i, value in enumerate(tup):
                random_val_in_the_patch = random.randint(0, s0)
                sparse_tup[i * (s0 + 1) + random_val_in_the_patch] = value  
            new_tuples.append(sparse_tup)

        new_tuples = torch.tensor(new_tuples, dtype=torch.int64).reshape(-1, m, sparse_tuple_size)

        old_feature_to_index = {e: i for i, e in enumerate(old_features)}
        old_paths_indices = [old_feature_to_index[f.item()] for f in old_paths]
        new_paths = new_tuples[old_paths_indices]

        all_levels_tuples.append(new_tuples)
        all_levels_paths.append(new_paths)

    return all_levels_paths, all_levels_tuples

def sample_hierarchical_rules_type_b(num_features, num_layers, m, num_classes, s, s0, seed=0):
    random.seed(seed)
    all_levels_paths = [torch.arange(num_classes)]
    all_levels_tuples = []
    
    for l in range(num_layers):
        old_paths = all_levels_paths[-1].flatten()
        old_features = list(set([i.item() for i in old_paths]))
        num_old_features = len(old_features)

        # Generate tuples with sparsity s(s_0 + 1) s informative s*s_0 uninformative
        sparse_tuple_size = s * (s0 + 1)
        possible_tuples = list(product(range(num_features), repeat=s))
        num_new_tuples = m * num_old_features

        random.shuffle(possible_tuples)
        selected_tuples = possible_tuples[:num_new_tuples]

        # sparse format 
        new_tuples = []
        for tup in selected_tuples:
            sparse_tup = [-1] * sparse_tuple_size  # Initialize as uninformative
            available_positions = list(range(sparse_tuple_size))  # All positions are available

            for value in tup:
                # Preserve the order of the values in the tuple.
                valid_positions = [pos for pos in available_positions if pos > (sparse_tup.index(value) if value in sparse_tup else -1)]
                
                position = random.choice(valid_positions)  
                sparse_tup[position] = value
                available_positions.remove(position)

            new_tuples.append(sparse_tup)


        new_tuples = torch.tensor(new_tuples, dtype=torch.int64).reshape(-1, m, sparse_tuple_size)

        old_feature_to_index = {e: i for i, e in enumerate(old_features)}
        old_paths_indices = [old_feature_to_index[f.item()] for f in old_paths]
        new_paths = new_tuples[old_paths_indices]

        all_levels_tuples.append(new_tuples)
        all_levels_paths.append(new_paths)

    return all_levels_paths, all_levels_tuples



def sample_data_from_paths(samples_indices, paths, m, num_classes, num_layers, s, s0, seed=0, seed_reset_layer=42):
    Pmax = m ** ((s ** num_layers - 1) // (s - 1)) * num_classes
    sparse_tuple_size = s * (s0 + 1)
    
    x = paths[-1].reshape(
        num_classes, *sum([(m, sparse_tuple_size) for _ in range(num_layers)], ())
    )  # [nc, m, sparse_tuple_size, ...]

    groups_size = Pmax // num_classes
    y = samples_indices.div(groups_size, rounding_mode='floor')
    samples_indices = samples_indices % groups_size

    indices = []
    for l in range(num_layers):
        if l != 0:
            left_right = (
                torch.arange(s)[None]
                .repeat(s ** (num_layers - 2), 1)
                .reshape(s ** (num_layers - l - 1), -1)
                .t()
                .flatten()
            )
            left_right = left_right[None].repeat(len(samples_indices), 1)
            indices.append(left_right)

        if l >= seed_reset_layer:
            np.random.seed(seed + 42 + l)
            perm = torch.randperm(len(samples_indices))
            samples_indices = samples_indices[perm]

        groups_size //= m ** (s ** l)
        layer_indices = samples_indices.div(groups_size, rounding_mode='floor')

        # Use sparsity-aware indexing for rules
        rules = number2base(layer_indices, m, string_length=s ** l)
        rules = (
            rules[:, None]
            .repeat(1, s ** (num_layers - l - 1), 1)
            .permute(0, 2, 1)
            .flatten(1)
        )
        indices.append(rules)

        samples_indices = samples_indices % groups_size

    yi = y[:, None].repeat(1, s ** (num_layers - 1))
    x = x[tuple([yi, *indices])].flatten(1)

    return x, y
