def random_attribution(image_tensor, dot_density=0.1):
    """
    Generate a random attribution salience map with colored dots.

    Args:
        image_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W).
        dot_density (float): Probability of a pixel being colored (0-1).
    """
    _, C, H, W = image_tensor.shape
    
   
    random_salience = torch.rand(H, W, device=image_tensor.device)
    

  
    random_salience =  torch.rand(H, W, device=image_tensor.device)
    
    return random_salience.cpu().numpy()


random_salience_map = random_attribution(input_tensor, dot_density=0.015)


original_image = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
