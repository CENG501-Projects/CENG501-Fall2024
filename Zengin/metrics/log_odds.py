import torch
import numpy as np

def compute_logoddsshift(
    model,
    explanation_method,
    x,
    num_steps=10,
    mask_value=0.0
):
    """
    Computes the average log-odds shift for the predicted class
    as we progressively remove top-salient pixels.
    
    Parameters
    ----------
    model : torch.nn.Module
        The pre-trained model in eval mode (e.g., ViT).
    explanation_method : callable
        A function that, given (model, x), returns a 2D salience map
        of shape (H, W) as a NumPy array or torch.Tensor in [0,1].
    x : torch.Tensor
        The input image tensor, shape (1, 3, H, W).
    num_steps : int
        Number of fractions from 0..1 to remove. e.g., num_steps=10 => remove 0%,10%,...,100%.
    mask_value : float
        Pixel value to fill in the “removed” region. Could be 0, or mean, or random noise.

    Returns
    -------
    avg_log_odds_shift : float
        The average log-odds shift across all fractions.
        Often negative if removing salient pixels lowers confidence.
    fractions : list
        The fractions used for removal (0..1).
    shifts : list
        The log-odds shift at each removal fraction.
    """
    model.eval()
    device = x.device

    
    with torch.no_grad():
        logits = model(x)  
        pred_class = logits.argmax(dim=1).item()
        
        probs = torch.softmax(logits, dim=1)
        p_original = probs[0, pred_class].item()

    
    eps = 1e-8  
    log_odds_original = np.log(p_original / (1 - p_original + eps) + eps)

    
    sal_map = explanation_method(model, x)
    if isinstance(sal_map, np.ndarray):
        sal_map = torch.from_numpy(sal_map).float().to(device)
    else:
        sal_map = sal_map.to(device).float()

    
    flat_sal = sal_map.view(-1)
    sorted_sal, sorted_idx = torch.sort(flat_sal, descending=True)

    fractions = np.linspace(0.0, 1.0, num_steps + 1)
    shifts = []  

  
    c, h, w = x.shape[1], x.shape[2], x.shape[3]
    for frac in fractions:
        num_remove = int(frac * len(sorted_sal))

      
        remove_idx = sorted_idx[:num_remove]

       
        x_perturbed = x.clone()

        
        x_flat = x_perturbed.view(1, c, h*w)

    
        for ch in range(c):
            x_flat[0, ch, remove_idx] = mask_value
        
        
        x_perturbed = x_flat.view(1, c, h, w)

      
        with torch.no_grad():
            logits_perturbed = model(x_perturbed)
            probs_perturbed = torch.softmax(logits_perturbed, dim=1)
            p_perturbed = probs_perturbed[0, pred_class].item()

      
        log_odds_perturbed = np.log(p_perturbed / (1 - p_perturbed + eps) + eps)

        
        shift = log_odds_perturbed - log_odds_original
        shifts.append(shift)

   
    avg_log_odds_shift = float(np.mean(shifts))

    return avg_log_odds_shift, fractions.tolist(), shifts




