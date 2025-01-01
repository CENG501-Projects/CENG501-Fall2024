import torch
import random

def apply_diffeomorphism(x, s, s0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        
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