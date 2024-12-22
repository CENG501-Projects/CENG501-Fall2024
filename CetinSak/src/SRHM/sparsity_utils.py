import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from itertools import *
import random
import numpy as np
from src.utils.utils import number2base
import torchshow as ts

def sample_hierarchical_rules_type_a(num_features, num_layers, m, num_classes, s, s0, seed=0):
    random.seed(seed)
    all_levels_paths = [torch.arange(num_classes)]
    all_levels_tuples = []
    
    for l in range(num_layers):
        old_paths = all_levels_paths[-1].flatten()
        old_features_set = set([i.item() for i in old_paths])
        old_features = list(old_features_set)
        num_old_features = len(old_features) 

        print("Num old features is:", num_old_features)

        # [0, 0] -> [-1, -1, 0 | -1, -1 0]

        # Generate tuples with sparsity s(s_0 + 1) s informative s*s_0 uninformative
        sparse_tuple_size = s * (s0 + 1)
        possible_tuples = list(product(range(num_features), repeat=s))
        num_new_tuples = m * num_old_features

        print("Num new tuples is:", num_new_tuples)
        print("Possible tuple num is:", len(possible_tuples))

        assert len(possible_tuples)*s*(s0+1) >= num_new_tuples, f"{num_features}**{s}={len(possible_tuples)} < {num_new_tuples}"

        random.shuffle(possible_tuples)
        selected_tuples = possible_tuples[:num_new_tuples]

        # [ 1,1 ,1,1, 1,1]

        # uninformative features are randomly placed in the patch here.
        new_tuples = []
        for tup in selected_tuples:
            sparse_tup = [-1] * sparse_tuple_size  
            for i, value in enumerate(tup):
                random_val_in_the_patch = random.randint(0, s0)
                sparse_tup[i * (s0 + 1) + random_val_in_the_patch] = value  
            new_tuples.append(sparse_tup)

        # In deeper layers, we have completely uninformative expansions
        print(f"Add {m} tuples for uninformative generations")
        for _ in range(m):
            sparse_tup = [-1] * sparse_tuple_size
            new_tuples.append(sparse_tup)

        print("Number of new tuples is:", len(new_tuples))

        new_tuples = torch.tensor(new_tuples, dtype=torch.int64).reshape(-1, m, sparse_tuple_size)

        old_feature_to_index = {e: i for i, e in enumerate(old_features)}
        old_paths_indices = [old_feature_to_index[f.item()] for f in old_paths]
        new_paths = new_tuples[old_paths_indices]

        print("Old feature to index is:", old_feature_to_index)
        print("Number of new paths is:", new_paths.shape)
        
        ts.save(new_paths, f"torchshow/paths_{l}_{s}_{s0}_{m}.png")

        all_levels_tuples.append(new_tuples)
        all_levels_paths.append(new_paths)

        print("")

    return all_levels_paths, all_levels_tuples

def sample_hierarchical_rules_type_b(num_features, num_layers, m, num_classes, s, s0, seed=0):
    import torch
    import random
    from itertools import product

    random.seed(seed)
    all_levels_paths = [torch.arange(num_classes)]
    all_levels_tuples = []
    
    for l in range(num_layers):
        old_paths = all_levels_paths[-1].flatten()
        old_features = list(set([i.item() for i in old_paths]))
        num_old_features = len(old_features)

        # Generate tuples with sparsity s(s_0 + 1): s informative and s * s0 uninformative
        sparse_tuple_size = s * (s0 + 1)
        possible_tuples = list(product(range(num_features), repeat=sparse_tuple_size))
        num_new_tuples = m * num_old_features

        # Validate sufficient tuples for the layer
        assert len(possible_tuples) >= num_new_tuples, "Not enough features to choose from with sparsity constraints!"

        random.shuffle(possible_tuples)
        selected_tuples = possible_tuples[:num_new_tuples]

        # Create sparse format
        new_tuples = []
        for tup in selected_tuples:
            sparse_tup = [-1] * sparse_tuple_size  # Initialize as uninformative
            available_positions = list(range(sparse_tuple_size))  # All positions are available
            
            # Ensure informative features maintain order
            for value in tup:
                # Find valid positions
                valid_positions = [
                    pos for pos in available_positions
                    if len([p for p in available_positions if p > pos]) >= (len(tup) - tup.index(value) - 1)
                ]
                assert valid_positions, "Not enough valid positions to maintain order!"
                
                position = random.choice(valid_positions)  # Choose a valid position
                sparse_tup[position] = value
                available_positions.remove(position)  # Update available positions

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

    paths_minus_uninformative = sum([(m, sparse_tuple_size) for _ in range(num_layers)], ())

    x = paths[-1].reshape(
        num_classes, *paths_minus_uninformative
    )  # [nc, m, sparse_tuple_size, ...]

    groups_size = Pmax // num_classes
    y = samples_indices.div(groups_size, rounding_mode='floor')
    samples_indices = samples_indices % groups_size

    indices = []
    for l in range(num_layers):
        if l != 0:
            left_right = (
                torch.arange(sparse_tuple_size)[None]
                .repeat(sparse_tuple_size ** (num_layers - 2), 1)
                .reshape(sparse_tuple_size ** (num_layers - l - 1), -1)
                .t()
                .flatten()
            )
            left_right = left_right[None].repeat(len(samples_indices), 1)
            indices.append(left_right)

        # Synonyms happen here (probably)
        if l >= seed_reset_layer:
            np.random.seed(seed + 42 + l)
            perm = torch.randperm(len(samples_indices))
            samples_indices = samples_indices[perm]

        print(f"Group size will be {groups_size}//({m} ** ({s} ** {l}))")
        groups_size //= m ** (s ** l)
        
        layer_indices = samples_indices.div(groups_size, rounding_mode='floor')

        # Use sparsity-aware indexing for rules
        rules = number2base(layer_indices, m, string_length=sparse_tuple_size ** l)
        rules = (
            rules[:, None]
            .repeat(1, sparse_tuple_size ** (num_layers - l - 1), 1)
            .permute(0, 2, 1)
            .flatten(1)
        )
        indices.append(rules)

        samples_indices = samples_indices % groups_size

    yi = y[:, None].repeat(1, sparse_tuple_size ** (num_layers - 1))
    x = x[tuple([yi, *indices])].flatten(1)

    return x, y
