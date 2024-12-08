import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import datetime
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from timm import create_model
import pickle

class RawAttention:
    def __init__(self, model, layer_index=-1):
        """
        Initialize Raw Attention visualization.
        
        Args:
            model: Vision Transformer model
            layer_index (int): Which transformer layer to visualize (-1 for last layer)
        """
        self.model = model.eval()
        self.layer_index = layer_index
        self.attention_maps = None
        self.handlers = []
        
        # Get the specified transformer block
        self.target_layer = self.model.blocks[layer_index]
        self._register_hooks()

    def _get_attention_hook(self, module, input, output):
        """Hook for capturing attention maps"""
        # The attention weights are stored in the module's qkv operation
        # We need to extract them after the softmax operation
        self.attention_maps = module.attn_drop.output  # Get attention weights after dropout

    def _register_hooks(self):
        """Register forward hook on the attention module"""
        def attention_hook(module, input, output):
            # Get input features
            x = input[0]
            B, N, C = x.shape
            
            # Get qkv weights
            qkv = module.qkv(x)
            head_dim = module.head_dim
            num_heads = module.num_heads
            
            # Reshape qkv
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            
            # Compute attention weights
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)  # [B, H, N, N]
            self.attention_maps = attn
        
        self.handlers.append(
            self.target_layer.attn.register_forward_hook(attention_hook)
        )

    def generate_attention_map(self, input_tensor):
        """
        Generate attention visualization for the input image.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor [1, 3, H, W]
            
        Returns:
            np.ndarray: Attention map resized to image dimensions
        """
        self.model.zero_grad()
        _ = self.model(input_tensor)
        
        # Get attention weights [B, H, N, N]
        attn_weights = self.attention_maps[0]  # Take first batch
        
        # Average over attention heads
        attn_weights = attn_weights.mean(0)  # [N, N]
        
        # Get attention weights for cls token to patch tokens
        patch_attn = attn_weights[0, 1:]  # [N-1]
        
        # Reshape to square grid
        grid_size = int(np.sqrt(patch_attn.size(0)))
        attention_map = patch_attn.reshape(grid_size, grid_size)
        
        # Resize to image size
        attention_map = F.interpolate(
            attention_map.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )
        
        # Convert to numpy and normalize
        attention_map = attention_map.squeeze().cpu().detach().numpy()
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        return attention_map

    def analyze_attention_patterns(self, input_tensor, num_regions=10):
        """
        Analyze attention patterns by dividing the image into regions.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor
            num_regions (int): Number of regions to analyze
            
        Returns:
            dict: Analysis results including attention map and regional scores
        """
        attention_map = self.generate_attention_map(input_tensor)
        
        # Flatten and sort attention values
        flat_attention = attention_map.flatten()
        sorted_indices = np.argsort(flat_attention)[::-1]
        
        # Divide into regions
        pixels_per_region = len(sorted_indices) // num_regions
        region_indices = [sorted_indices[i:i + pixels_per_region] 
                         for i in range(0, len(sorted_indices), pixels_per_region)]
        
        # Compute attention scores for each region
        region_scores = []
        for indices in region_indices:
            score = np.sum(flat_attention[indices])
            region_scores.append(float(score))
            
        return {
            'attention_map': attention_map,
            'region_scores': np.array(region_scores),
            'region_indices': region_indices
        }

    def perturb_image(self, input_image, subset_indices, strategy="mean"):
        """
        Perturb an input image by masking specific pixels.
        
        Args:
            input_image (torch.Tensor): Input image tensor (B, C, H, W)
            subset_indices (np.ndarray): Indices of pixels to perturb
            strategy (str): Perturbation strategy ("mean", "black", or "random")
            
        Returns:
            torch.Tensor: Perturbed image
        """
        device = input_image.device
        perturbed_image = input_image.clone()
        
        # Convert flat indices to 2D coordinates
        h, w = input_image.shape[2:]
        rows, cols = np.unravel_index(subset_indices, (h, w))
        
        # Define ImageNet normalization parameters
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        if strategy == "mean":
            mask_value = imagenet_mean.view(3, 1).expand(3, len(rows))
        elif strategy == "black":
            mask_value = torch.zeros(3, len(rows), device=device)
        elif strategy == "random":
            mask_value = torch.randn(3, len(rows), device=device)
        else:
            raise ValueError(f"Unknown perturbation strategy: {strategy}")
        
        # Apply mask
        perturbed_image[0, :, rows, cols] = mask_value
        
        return perturbed_image

    def analyze_perturbation_impact(self, input_image, target_class=None, num_regions=10, 
                                  strategy="mean"):
        """
        Analyze the impact of perturbations on model predictions.
        
        Args:
            input_image (torch.Tensor): Input image tensor
            target_class (int, optional): Target class index
            num_regions (int): Number of regions for analysis
            strategy (str): Perturbation strategy
            
        Returns:
            dict: Analysis results including attention map, scores, and confidence changes
        """
        # Generate attention map
        attention_map = self.generate_attention_map(input_image)
        
        # Get original prediction and confidence
        with torch.no_grad():
            original_output = self.model(input_image)
            if target_class is None:
                target_class = original_output.argmax(dim=1).item()
            original_confidence = F.softmax(original_output, dim=1)[0, target_class].item()
        
        # Compute region scores and get indices
        flat_attention = attention_map.flatten()
        sorted_indices = np.argsort(flat_attention)[::-1]
        
        # Divide into regions
        pixels_per_region = len(sorted_indices) // num_regions
        region_indices = [sorted_indices[i:i + pixels_per_region] 
                         for i in range(0, len(sorted_indices), pixels_per_region)]
        
        # Analyze perturbation impact
        confidence_changes = []
        region_scores = []
        
        for indices in region_indices:
            # Compute region score
            score = np.sum(flat_attention[indices])
            region_scores.append(float(score))
            
            # Perturb image
            perturbed_image = self.perturb_image(input_image, indices, strategy)
            
            # Compute new confidence
            with torch.no_grad():
                output = self.model(perturbed_image)
                confidence = F.softmax(output, dim=1)[0, target_class].item()
                confidence_change = original_confidence - confidence
                confidence_changes.append(confidence_change)
        
        return {
            'attention_map': attention_map,
            'region_scores': np.array(region_scores),
            'confidence_changes': np.array(confidence_changes),
            'region_indices': region_indices
        }

    def __del__(self):
        """Clean up by removing hooks"""
        for handler in self.handlers:
            handler.remove()

def visualize_attention(image_tensor, attention_map, save_path):
    """
    Visualize the original image, attention map, and overlay.
    
    Args:
        image_tensor (torch.Tensor): Original image tensor
        attention_map (np.ndarray): Generated attention map
        save_path (str): Path to save the visualization
    """
    # Convert image tensor to numpy array
    image = image_tensor.permute(1, 2, 0).numpy()
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Attention map
    plt.subplot(1, 3, 2)
    plt.imshow(attention_map, cmap='jet')
    plt.title('Attention Map')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(attention_map, cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Setup paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_dir, 'ViT-L-16_cifar10.pth')
    indices_path = os.path.join(project_dir, 'cifar10_eval_indices.pkl')
    
    # Create results directories
    results_dir = os.path.join(project_dir, 'results')
    attention_dir = os.path.join(results_dir, 'salience_maps', 'cifar10', 'raw_attention')
    scores_dir = os.path.join(results_dir, 'salience_scores', 'raw_attention')
    
    os.makedirs(attention_dir, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)
    
    # Create additional directory for perturbation results
    perturb_dir = os.path.join(results_dir, 'perturbation_analysis', 'raw_attention')
    os.makedirs(perturb_dir, exist_ok=True)
    
    # Define ImageNet mean and std for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('vit_large_patch16_224', pretrained=False, num_classes=10)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # Initialize raw attention visualizer
    raw_attention = RawAttention(model)
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    with open(indices_path, 'rb') as f:
        eval_indices = pickle.load(f)
    dataset = torch.utils.data.Subset(dataset, eval_indices)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Process images
    for i in range(10):
        image, label = dataset[i]
        image = image.unsqueeze(0).to(device)
        
        # Get model prediction
        with torch.no_grad():
            pred = model(image).argmax(dim=1).item()
        
        # Generate attention visualization and analysis with perturbation
        results = raw_attention.analyze_perturbation_impact(
            image,
            target_class=None,  # Use predicted class
            num_regions=10,
            strategy="mean"
        )
        
        # Save visualization
        filename = f'attention_img{i:02d}_{class_names[label]}_pred_{class_names[pred]}.png'
        save_path = os.path.join(attention_dir, filename)
        visualize_attention(image.squeeze(0).cpu(), results['attention_map'], save_path)
        
        # Save perturbation visualization
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        img_display = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)
        plt.imshow(img_display)
        plt.title('Original Image')
        plt.axis('off')
        
        # Attention map
        plt.subplot(1, 3, 2)
        plt.imshow(results['attention_map'], cmap='jet')
        plt.title('Attention Map')
        plt.axis('off')
        
        # Perturbed image (using most important region)
        plt.subplot(1, 3, 3)
        perturbed_image = raw_attention.perturb_image(
            image,
            results['region_indices'][0],  # Most important region
            strategy="mean"
        )
        img_display = perturbed_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)
        plt.imshow(img_display)
        plt.title('Perturbed Image\n(Most Important Region)')
        plt.axis('off')
        
        plt.tight_layout()
        perturb_save_path = os.path.join(perturb_dir, f'perturbation_analysis_img{i:02d}.png')
        plt.savefig(perturb_save_path)
        plt.close()
        
        # Save analysis results
        results_dict = {
            'image_index': i,
            'true_class': class_names[label],
            'predicted_class': class_names[pred],
            'region_scores': results['region_scores'].tolist(),
            'confidence_changes': results['confidence_changes'].tolist(),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        scores_file = os.path.join(scores_dir, f'attention_scores_img{i:02d}.json')
        with open(scores_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print(f'Processed image {i+1}/10: {filename}')

if __name__ == '__main__':
    main()
