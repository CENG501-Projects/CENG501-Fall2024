# This file will hold the main implementation of SHRM.
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchshow as ts

import copy
from itertools import *
import random
import numpy as np

from src.SRHM.sparsity_utils import *
from src.utils.utils import *
from src.SRHM.diffeomorphism_utilities import *
import tempfile
import os

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
            apply_diffeo=False,
            train=True,
            input_format='onehot',
            whitening=0,
            transform=None,
            testsize=-1,
            seed_reset_layer=42,
            seed_p=None):
        
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
        self.apply_diffeo = apply_diffeo
        self.seed_p = seed_p

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

        seed_synonym = self.seed
        self.seed_synonym = seed_synonym
        seed_diffeo = self.seed
        if self.seed_p is not None:
            seed_diffeo = self.seed
            seed_synonym = self.seed_p
            self.seed_synonym = seed_synonym
        
        ## Generate the dataset
        paths, _ = self.sample_hierarchical_rules(
            self.num_features, self.num_layers, self.m, self.num_classes, self.s, self.s0, self.seed, seed_diffeo
        )

        self.paths = paths

        ## Check Pmax calculation.
        Pmax = self.m ** ((self.s ** self.num_layers - 1) // (self.s - 1)) * self.num_classes
        assert Pmax < 1e19 
        if max_dataset_size is None or max_dataset_size > Pmax:
            max_dataset_size = Pmax
        if testsize == -1:
            testsize = min(max_dataset_size // 5, 5000)

        g = torch.Generator()
        g.manual_seed(seed_traintest_split)


        if Pmax < 5e6:  # there is a crossover in computational time of the two sampling methods around this value of Pmax
            samples_indices = torch.randperm(Pmax, generator=g)[:max_dataset_size]
        else:
            samples_indices = torch.randint(Pmax, (2 * max_dataset_size,), generator=g)
            samples_indices = torch.unique(samples_indices)
            perm = torch.randperm(len(samples_indices), generator=g)[:max_dataset_size]
            samples_indices = samples_indices[perm]

        if len(samples_indices) > 2e5:
            samples_indices = samples_indices[:int(2e5)]

        if train and testsize:
            samples_indices = samples_indices[:-testsize]
        else:
            samples_indices = samples_indices[-testsize:]

        self.samples_indices = samples_indices

        if len(self.samples_indices) > 1e6:
            print("Data is too big, will calculate each idx separately")
            return

        # take sample to infer shape
        x, targets = self.sample_data_from_paths(
            samples_indices[:2], paths, m, num_classes, num_layers, s, s0=s0, seed=seed_synonym, synonym_start_layer=seed_reset_layer
        )
        x = x + 1
        x = self.transform_x(x)

        chunk_size = 1000
        cache_dir = ".cache"
        self.temp_dir = None
        if cache_dir is None:
            self.temp_dir = tempfile.mkdtemp()
            cache_dir = self.temp_dir

        self.data_path = os.path.join(cache_dir, f'data_{train}_{num_layers}_{s0}_{m}_{seed_synonym}_{seed_diffeo}.mmap')
        first_dim = len(samples_indices)  # First dimension from sample_indices
        other_dims = x.shape[1:]  # Other dimensions from x
        final_shape = (first_dim,) + other_dims  # Combine to get the final shape
        print(f"Final shape is {final_shape}")
        self.dataset_size = len(samples_indices)

        self.targets = []

        fp = np.memmap(self.data_path, dtype='float32', mode='w+', 
                shape=final_shape)
        for i in range(0, len(samples_indices), chunk_size):
            # Get the current chunk
            chunk = samples_indices[i:i + chunk_size]
            x, targets = self.sample_data_from_paths(
                chunk, paths, m, num_classes, num_layers, s, s0=s0, seed=seed_synonym, synonym_start_layer=seed_reset_layer
            )
            x = x + 1
            x = self.transform_x(x)
            x_np = x.numpy()

            start_index = i
            end_index = i+len(chunk)
            fp[start_index:end_index] = x_np[:]
            fp.flush()
            
            del x, x_np

            self.targets.append(targets)

        del fp
        # Create memory-mapped arrays
        self.data = np.memmap(self.data_path, dtype='float32', mode='r',
                             shape=final_shape)

        self.targets = torch.cat(self.targets)

        self.transform = transform

    def transform_x(self, x):
        if "pairs" in self.input_format:
            x = pairing_features(x, self.num_features)

        if 'onehot' not in self.input_format:
            assert not self.whitening, "Whitening only implemented for one-hot encoding"

        if "binary" in self.input_format:
            x = dec2bin(x)
            x = x.permute(0, 2, 1)
        elif "long" in self.input_format:
            x = x.long() + 1
        elif "decimal" in self.input_format:
            x = ((x[:, None] + 1) / self.num_features - 1) * 2
        elif "onehot" in self.input_format:
            x = F.one_hot(
                x.long(),
                num_classes=self.num_features+1 if 'pairs' not in self.input_format else self.num_features ** 2
            ).float()
            x = x.permute(0, 2, 1)

            if self.whitening:
                inv_sqrt_n = (self.num_features - 1) ** -.5
                x = x * (1 + inv_sqrt_n) - inv_sqrt_n

        else:
            raise ValueError
    
        return x
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

        print("Shape of the SRHM:", self.x.shape)
        print("An example:", x)

    def __len__(self):
        return len(self.samples_indices)

    def __getitem__(self, idx):
        """
        :param idx: sample index
        :return (torch.tensor, torch.tensor): (sample, label)
        """
        if len(self.samples_indices) > 16:
            x, targets = self.sample_data_from_paths(
                torch.atleast_1d(self.samples_indices[idx]), self.paths, self.m, self.num_classes, self.num_layers, self.s, s0=self.s0, seed=self.seed_synonym, synonym_start_layer=self.seed_reset_layer
            )

            x = x + 1
            x = self.transform_x(x)
            x = x[0]
            targets = targets[0]

            return x, targets

        x = torch.from_numpy(self.data[idx])
        y = self.targets[idx]

        if self.transform:
            x, y = self.transform(x, y)


        if self.apply_diffeo:
            wodiffeo_x = copy.deepcopy(x)
            x = apply_diffeomorphism(x, self.s, self.s0, seed=idx)

            return x,wodiffeo_x, y

        # if self.background_noise:
        #     g = torch.Generator()
        #     g.manual_seed(idx)
        #     x += torch.randn(x.shape, generator=g) * self.background_noise

        return x, y