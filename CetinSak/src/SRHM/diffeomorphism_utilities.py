import torch
import random

def apply_srhm_synonym(x, s, s0, num_layers, seed=0):

    torch.manual_seed(seed) 

    batch_size, num_features, feature_dim = x.shape

    patch_size = s * (s0 + 1)

    if feature_dim % patch_size != 0:
        raise ValueError(
            f"Adjust s or s0 to match."
        )

    num_patches = feature_dim // patch_size

    x_transformed = x.clone()

    for b in range(batch_size):
        for f in range(num_features):
            for p in range(num_patches):
                patch_start = p * patch_size
                patch_end = patch_start + patch_size
                patch = x[b, f, patch_start:patch_end]

                informative_positions = torch.arange(0, patch_size, step=(s0 + 1))

                informative_features = patch[informative_positions]
                perm = torch.randperm(len(informative_features))  # burada synonym replacement olcak. gibi?
                informative_features = informative_features[perm]

                patch[informative_positions] = informative_features

                x_transformed[b, f, patch_start:patch_end] = patch

    return x_transformed


def apply_srhm_diffeomorphism(x, s, s0):
    batch_size, num_features, feature_dim = x.shape

    patch_size = s * (s0 + 1)

    if feature_dim % patch_size != 0:
        raise ValueError(
            f"Adjust s or s0 to match."
        )

    num_patches = feature_dim // patch_size

    x_transformed = x.clone()

    for b in range(batch_size):
        for f in range(num_features):
            for p in range(num_patches):
                patch_start = p * patch_size
                patch_end = patch_start + patch_size
                patch = x[b, f, patch_start:patch_end]

                informative_positions = torch.arange(0, patch_size, step=(s0 + 1))
                uninformative_positions = torch.tensor(
                    [i for i in range(patch_size) if i not in informative_positions],
                    dtype=torch.long,
                    device=x.device
                )

                informative_features = patch[informative_positions]
                perm = torch.randperm(len(informative_features))
                informative_features = informative_features[perm]

                patch[informative_positions] = informative_features
                patch[uninformative_positions] = patch[uninformative_positions]

                x_transformed[b, f, patch_start:patch_end] = patch

    return x_transformed

def apply_diffeomorphism(x, s, s0):
        
    batch_size = x.shape[0]
    patch_size = s * (s0 + 1)
    num_patches = x.shape[1] // patch_size
    
    transformed = torch.zeros_like(x)
    
    for b in range(batch_size):
        for p in range(num_patches):
            patch = x[b, p*patch_size:(p+1)*patch_size]
            
            informative_positions = torch.nonzero(patch).squeeze()
            
            if len(informative_positions.shape) == 0: 
                if informative_positions.numel() > 0: 
                    informative_positions = informative_positions.unsqueeze(0)
                else:
                    continue
                    
            if len(informative_positions) == 0:
                transformed[b, p*patch_size:(p+1)*patch_size] = patch
                continue
                
            informative_values = patch[informative_positions]
            
            new_positions = generate_valid_positions()
            
            new_patch = torch.zeros_like(patch)
            new_patch[new_positions] = informative_values
            
            transformed[b, p*patch_size:(p+1)*patch_size] = new_patch
            
    return transformed

def generate_valid_positions():
    ## TODO: Function to generates valid positions to the informative features,here.
    
    pass


## Checker.
def check_order_preservation(original, transformed, s, s0):
    patch_size = s * (s0 + 1)
    num_patches = original.shape[1] // patch_size
    
    for b in range(original.shape[0]):
        for p in range(num_patches):
            orig_patch = original[b, p*patch_size:(p+1)*patch_size]
            trans_patch = transformed[b, p*patch_size:(p+1)*patch_size]
            
            orig_positions = torch.nonzero(orig_patch).squeeze()
            trans_positions = torch.nonzero(trans_patch).squeeze()
            
            if len(orig_positions.shape) == 0:
                if orig_positions.numel() > 0:
                    orig_positions = orig_positions.unsqueeze(0)
                    trans_positions = trans_positions.unsqueeze(0)
                else:
                    continue
                    
            if len(orig_positions) == 0:
                continue
                
            orig_values = orig_patch[orig_positions]
            trans_values = trans_patch[trans_positions]
            
            if not torch.all(orig_values == trans_values):
                return False
                
    return True