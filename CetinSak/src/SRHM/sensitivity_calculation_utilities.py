## This file will hold the sensitivity calculations for SRHM.

import torch
import torch.nn as nn

def calculate_diffeo_sensitivity(model, dataloader, s, s0, layer_idx=None, device='cuda', num_diffeos=5):
    model.eval()
    model = model.to(device)

    numerator = 0.0  # ||fk(x) - fk(τ(x))||²
    denominator = 0.0  # ||fk(x₁) - fk(x₂)||²
    
    for batch_idx, (x,x_with_diffeo,_) in enumerate(dataloader):
        x = x.to(device)
        batch_size = x.shape[0]
        
        # Calculate original outputs/representations
        with torch.no_grad():
            orig_output = model.get_layer_output(x)
            transformed_output = model.get_layer_output(x_with_diffeo)
        

        numerator = (orig_output - transformed_output).pow(2).mean(dim=1)

        # Calculate baseline variation between different samples here....
        # ..............
        
        num_samples += batch_size
        
        if num_samples >= 1000:
            break
    
    numerator /= num_samples
    denominator /= (num_samples * (num_samples - 1) / 2)
    
    sensitivity = numerator / denominator
    return sensitivity