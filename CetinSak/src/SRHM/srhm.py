# This file will hold the main implementation of SHRM.
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchshow as ts

from itertools import *
import random
import numpy as np

from src.SRHM.sparsity_utils import *
from src.utils.utils import *

class SparseRandomHierarchyModel(Dataset):
    """
    Implement the Sparse Random Hierarchy Model (SRHM) as a PyTorch dataset.
    """

    def __init__(
            self,
            num_features=8,
            m=2,  # features multiplicity
            num_layers=2,
            num_classes=2,
            s=2,
            s0 =1,
            sparsity_type='a',
            seed=0,
            max_dataset_size=None,
            seed_traintest_split=0,
            train=True,
            input_format='onehot',
            whitening=0,
            transform=None,
            testsize=-1,
            seed_reset_layer=42,):
        
        torch.manual_seed(seed)
        self.num_features = num_features
        self.m = m
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.s = s
        self.s0 = s0
        self.seed = seed
        self.max_dataset_size = max_dataset_size
        self.seed_traintest_split = seed_traintest_split
        self.train = train
        self.input_format = input_format
        self.whitening = whitening
        self.transform = transform
        self.testsize = testsize
        self.seed_reset_layer = seed_reset_layer
        
        ## Set sparsity functions
        if sparsity_type == 'a':
            print("SRHM: Using sparsity type A")
            self.sample_hierarchical_rules = sample_hierarchical_rules_type_a
        elif sparsity_type == 'b':
            print("SRHM: Using sparsity type B")
            self.sample_hierarchical_rules = sample_hierarchical_rules_type_b
        ## Many other sparsity strategies here...
        else:
            raise ValueError("Sparsity type not recognized")

        self.sample_data_from_paths = sample_data_from_paths
        
        ## Generate the dataset
        paths, _ = self.sample_hierarchical_rules(
            self.num_features, self.num_layers, self.m, self.num_classes, self.s, self.s0, self.seed
        )

        ## Check Pmax calculation.
        Pmax = self.m ** ((self.s ** self.num_layers - 1) // (self.s - 1)) * self.num_classes
        assert Pmax < 1e19 
        if max_dataset_size is None or max_dataset_size > Pmax:
            max_dataset_size = Pmax
        if testsize == -1:
            testsize = min(max_dataset_size // 5, 20000)

        g = torch.Generator()
        g.manual_seed(seed_traintest_split)


        if Pmax < 5e6:  # there is a crossover in computational time of the two sampling methods around this value of Pmax
            samples_indices = torch.randperm(Pmax, generator=g)[:max_dataset_size]
        else:
            samples_indices = torch.randint(Pmax, (2 * max_dataset_size,), generator=g)
            samples_indices = torch.unique(samples_indices)
            perm = torch.randperm(len(samples_indices), generator=g)[:max_dataset_size]
            samples_indices = samples_indices[perm]

        if train and testsize:
            samples_indices = samples_indices[:-testsize]
        else:
            samples_indices = samples_indices[-testsize:]

        self.x, self.targets = self.sample_data_from_paths(
            samples_indices, paths, m, num_classes, num_layers, s, s0=s0, seed=seed, seed_reset_layer=seed_reset_layer
        )

        mode = "test"
        if train:
            mode = "train"
        
        grouped_examples = {label: [] for label in self.targets.unique().tolist()}
        for label, example in zip(self.targets, self.x):
            grouped_examples[label.item()].append(example)

        for label, ex_list in grouped_examples.items():
            print(f"Label: {label}")
            ts.save(torch.stack(ex_list, dim=0), f"torchshow/categorized_{self.num_layers}_{self.s}_{self.s0}_{self.m}_{mode}_{label}.png")

        ts.save(self.x, f"torchshow/examples_{self.num_layers}_{self.s}_{self.s0}_{self.m}_{mode}.png")
        self.x = self.x+1
        print(f"There are {len(self.x)} examples")
        print(f"There are {len(self.targets)} labels")

        self.sanity_check_sparsity(self.x[0])
        

        # encode input pairs instead of features
        if "pairs" in input_format:
            self.x = pairing_features(self.x, num_features)

        if 'onehot' not in input_format:
            assert not whitening, "Whitening only implemented for one-hot encoding"

        if "binary" in input_format:
            self.x = dec2bin(self.x)
            self.x = self.x.permute(0, 2, 1)
        elif "long" in input_format:
            self.x = self.x.long() + 1
        elif "decimal" in input_format:
            self.x = ((self.x[:, None] + 1) / num_features - 1) * 2
        elif "onehot" in input_format:
            self.x = F.one_hot(
                self.x.long(),
                num_classes=num_features+1 if 'pairs' not in input_format else num_features ** 2
            ).float()
            self.x = self.x.permute(0, 2, 1)

            if whitening:
                inv_sqrt_n = (num_features - 1) ** -.5
                self.x = self.x * (1 + inv_sqrt_n) - inv_sqrt_n

        else:
            raise ValueError

        self.transform = transform

    def sanity_check_sparsity(self, x):
        print(f"Feature/all ratio in example: %{(torch.count_nonzero(x)/torch.numel(x))*100}")
        print(f"Feature/all in example: {torch.count_nonzero(x)}/{torch.numel(x)}")

        expected_meaningful = self.s ** self.num_layers
        expected_total = (self.s * (self.s0 + 1)) ** self.num_layers
        print(f"Expected feature/all: {expected_meaningful}/{expected_total}")

        assert expected_meaningful == torch.count_nonzero(x), "Expected number of meaningful features do not match with actual."
        assert expected_total == torch.numel(x), "Expected length of last layer vector does not match with the actual."

        targets_unique = torch.unique(self.targets, return_counts=True) 
        class_count = len(targets_unique[0].tolist())
        print(f"Expected class count: {class_count}")
        print(f"Actual class count: {self.num_classes}")

        assert class_count == self.num_classes, "Expected class count and actual class count does not match."

        counts = targets_unique[1].float()
        counts = counts/counts.sum()
        print(f"Class distribution : - {torch.unique(self.targets, return_counts=True)} - {counts}")

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