# This file will hold the main implementation of SHRM.
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from itertools import *
import random
import numpy as np

class SparseRandomHierarchyModel(Dataset):
    """
    Implement the Sparse Random Hierarchy Model (SRHM) as a PyTorch dataset.
    """

    def __init__(self,
            num_features=8,
            m=2,  # features multiplicity
            num_layers=2,
            num_classes=2,
            s=2,
            seed=0,
            max_dataset_size=None,
            seed_traintest_split=0,
            train=True,
            input_format='onehot',
            whitening=0,
            transform=None,
            testsize=-1,
            seed_reset_layer=42,):
        pass

    def sample_hierarchical_rules():
        """
        Build hierarchy of features.
        :param num_features: number of features to choose from at each layer (short: `n`).
        :param num_layers: number of layers in the hierarchy (short: `l`)
        :param m: features multiplicity (number of ways in which a feature can be made from sub-feat.)
        :param num_classes: number of different classes
        :param s: sub-features tuple size
        :param seed: sampling sub-features seed
        :return: features hierarchy as a list of length num_layers.
                Each layer contains all paths going from label to layer.
        """
        random.seed(seed)
        all_levels_paths = [torch.arange(num_classes)]
        all_levels_tuples = []
        for l in range(num_layers):
            old_paths = all_levels_paths[-1].flatten()
            # unique features in the previous level
            old_features = list(set([i.item() for i in old_paths]))
            num_old_features = len(old_features)
            # new_features = list(combinations(range(num_features), 2))
            # generate all possible new features at this level
            new_tuples = list(product(*[range(num_features) for _ in range(s)]))
            assert (
                    len(new_tuples) >= m * num_old_features
            ), "Not enough features to choose from!!"
            random.shuffle(new_tuples)
            # samples as much as needed
            new_tuples = new_tuples[: m * num_old_features]
            new_tuples = list(sum(new_tuples, ()))  # tuples to list

            new_tuples = torch.tensor(new_tuples)

            new_tuples = new_tuples.reshape(-1, m, s)  # [n_features l-1, m, 2]

            # next two lines needed because not all features are necessarily samples in previous level
            old_feature_to_index = dict([(e, i) for i, e in enumerate(old_features)])
            old_paths_indices = [old_feature_to_index[f.item()] for f in old_paths]

            new_paths = new_tuples[old_paths_indices]

            all_levels_tuples.append(new_tuples)
            all_levels_paths.append(new_paths)

        return all_levels_paths, all_levels_tuples

    def sample_data_from_paths():
        pass

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        """
        :param idx: sample index
        :return (torch.tensor, torch.tensor): (sample, label)
        """

        x, y = self.x[idx], self.targets[idx]

        if self.transform:
            x, y = self.transform(x, y)

        # if self.background_noise:
        #     g = torch.Generator()
        #     g.manual_seed(idx)
        #     x += torch.randn(x.shape, generator=g) * self.background_noise

        return x, y