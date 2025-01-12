def random_attribution(image_tensor, dot_density=0.1):
    """
    Generate a random attribution salience map with colored dots.

    Args:
        image_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W).
        dot_density (float): Probability of a pixel being colored (0-1).
    """
    _, C, H, W = image_tensor.shape
    
    # Generate random values
    random_salience = torch.rand(H, W, device=image_tensor.device)
    
    # Create binary mask for dots
    
    
    # Give random values to the dots
    random_salience =  torch.rand(H, W, device=image_tensor.device)
    
    return random_salience.cpu().numpy()

# Generate random attribution
random_salience_map = random_attribution(input_tensor, dot_density=0.015)

# Convert input tensor to image for visualization
original_image = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

# Single plot with overlay
plt.figure(figsize=(6, 6))
plt.imshow(original_image, alpha=0.7)
plt.imshow(random_salience_map, cmap='jet', alpha=0.4)
plt.axis('off')
plt.show()