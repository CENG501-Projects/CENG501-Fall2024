import torch
import numpy as np

def compute_AOPC(
    model,
    explanation_method,
    x,
    num_steps=10,
    mask_value=0.0,
):
    """
    Compute the Area-Over-the-Perturbation-Curve (AOPC).
    
    Parameters
    ----------
    model : torch.nn.Module
        The pre-trained model in eval mode (e.g., ViT).
    explanation_method : callable
        A function that, given (model, x), returns a salience map 
        of shape (H, W) as a NumPy array or torch.Tensor in [0, 1].
    x : torch.Tensor
        The input image tensor, shape (1, 3, H, W).
    num_steps : int
        Number of equally spaced steps from 0% to 100% of salient pixels removed.
    mask_value : float
        The value used to replace removed pixels (0.0, or mean, or random noise).
    
    Returns
    -------
    AOPC : float
        The computed Area Over the Perturbation Curve.
    """
    model.eval()
    device = x.device

    
    with torch.no_grad():
        output = model(x)  
        pred_class = output.argmax(dim=1).item()
        orig_score = output[0, pred_class].item()  
    
    
    sal_map = explanation_method(model, x)  
    if isinstance(sal_map, np.ndarray):
        sal_map = torch.from_numpy(sal_map).float().to(device)
    else:
        sal_map = sal_map.to(device).float()
    
  

    flat_sal = sal_map.view(-1)              
    sorted_sal, sorted_idx = torch.sort(flat_sal, descending=True)

  
    fractions = torch.linspace(0, 1, steps=num_steps + 1)  
  
    
  
    drops = []

    
    for f in fractions:
        frac_val = f.item()
       
        num_remove = int(frac_val * len(sorted_sal))

       
        remove_idx = sorted_idx[:num_remove]

      
        x_perturbed = x.clone()

  
        c, h, w = x_perturbed.shape[1:]
        x_flat = x_perturbed.view(1, c, h * w) 

       
        for ch in range(c):
            x_flat[0, ch, remove_idx] = mask_value

        
        x_perturbed = x_flat.view(1, c, h, w)

       
        with torch.no_grad():
            out_perturbed = model(x_perturbed)
            new_score = out_perturbed[0, pred_class].item()
        
       
        drop = orig_score - new_score
        drops.append(drop)
    
.

    mean_drop = sum(drops) / len(drops)  # average drop
    return mean_drop
