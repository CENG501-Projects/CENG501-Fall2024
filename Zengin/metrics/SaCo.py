import numpy as np
import torch

def compute_SaCo(model, explanation_method, x, K=5):
    """
    Salience-guided Faithfulness Coefficient (SaCo).

    Parameters:
    -----------
    - model              : Pre-trained model (e.g., ViT).
    - explanation_method : A function or callable that given (model, x) returns a salience map.
    - x                  : Input image tensor, shape (1, 3, 224, 224).
    - K                  : Number of groups G_i.

    Returns:
    --------
    - F: Faithfulness coefficient (float).
    """

 
    salience_map = explanation_method(model, x)
    if isinstance(salience_map, np.ndarray):
        salience_map = torch.from_numpy(salience_map).float().to(x.device)
    elif isinstance(salience_map, torch.Tensor):
        salience_map = salience_map.to(x.device).float()
    else:
        raise TypeError("salience_map must be a NumPy array or torch.Tensor.")

  
    G_list = []
    s_G = []
    pred_x_G = []

    for i in range(K):
        
        G_i = generate_perturbation(x, salience_map, i, K)
        G_list.append(G_i)

       
        s_val = compute_salience_quantity(x, G_i, salience_map)
        s_G.append(s_val)

        
        
        pred_val = compute_prediction_influence(model, G_i)
        pred_x_G.append(pred_val)

   
    F = 0.0
    totalWeight = 0.0

    for i in range(K - 1):
        for j in range(i + 1, K):
            if pred_x_G[i] >= pred_x_G[j]:
                weight = s_G[i] - s_G[j]
            else:
                weight = -(s_G[i] - s_G[j])

            F += weight
            totalWeight += abs(weight)

    if totalWeight == 0:
        raise ValueError("The total weight is zero (no differences among perturbations).")
    else:
        F /= totalWeight

    return F



def generate_perturbation(original_tensor, salience_map, i, K):
    """
    Create a perturbation G_i from the original image x, guided by salience_map.
    For example, we can:
     - compute a threshold to mask out top (i+1)/K fraction of salient pixels
     - zero them out or add random noise
    """
    perturbed = original_tensor.clone()

    
    flat_sal = salience_map.view(-1)
    fraction = float(i + 1) / K
    sorted_sal, _ = torch.sort(flat_sal, descending=True)

    cutoff_index = int(len(sorted_sal) * fraction)
    cutoff_index = min(cutoff_index, len(sorted_sal) - 1)  
    cutoff_value = sorted_sal[cutoff_index].item()

    
    mask = (salience_map >= cutoff_value).float()

   
    mask_4d = mask.unsqueeze(0).unsqueeze(0)
    perturbed = perturbed * (1 - mask_4d)

    return perturbed


def compute_salience_quantity(original_tensor, perturbed_tensor, salience_map):
    """
    Example: s(G_i) = sum of salience in the region that changed 
    between original and perturbed.
    """
    
    diff = (original_tensor - perturbed_tensor).abs().sum(dim=1, keepdim=True)
    changed_mask = (diff > 1e-7).float()

    
    sal_4d = salience_map.unsqueeze(0).unsqueeze(0)
    changed_salience = changed_mask * sal_4d
    return changed_salience.sum().item()


def compute_prediction_influence(model, perturbed_tensor):
    """
    For demonstration: returns the model's top logit on `perturbed_tensor`.
    i.e., ∇pred(x, G_i) is approximated by the maximum logit of model(G_i).
    """
    with torch.enable_grad():
        out = model(perturbed_tensor)
        top_logit = out.max(dim=1)[0]
        return top_logit.item()
