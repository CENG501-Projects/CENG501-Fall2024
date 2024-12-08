import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import numpy as np
import os
import pickle
import cv2
from PIL import Image
import json
import datetime
from timm import create_model
import matplotlib.pyplot as plt

def show_cam_on_image(img, mask):
    """Create heatmap from mask on image"""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    """
    Compute rollout attention following the paper's implementation.
    Only use attention to CLS token and handle residual connections properly.
    """
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    
    # Add residual connections at each layer
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    
    # Normalize with softmax
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                    for i in range(len(all_layer_matrices))]
    
    # Multiply attention matrices in reverse order
    joint_attention = matrices_aug[-1]
    for i in range(len(matrices_aug)-2, -1, -1):
        joint_attention = joint_attention.bmm(matrices_aug[i])
    
    # Extract attention from CLS token to patches
    cls_attention = joint_attention[:, 0, 1:]  # [batch_size, num_patches]
    
    return cls_attention

class TransformerAttribution:
    def __init__(self, model):
        self.model = model.eval()
        self.attention_maps = {}
        self.handlers = []
        self._register_hooks()
        self.imagenet_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if "attn" in name and "blocks" in name and not any(x in name for x in ['q_norm', 'k_norm', 'proj', 'drop']):
                self.handlers.append(
                    module.register_forward_hook(self._attention_hook(name))
                )

    def _attention_hook(self, name):
        def hook(module, input, output):
            if hasattr(module, 'qkv'):
                qkv = module.qkv(input[0])
                qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * module.scale
                attn = attn.softmax(dim=-1)
                self.attention_maps[name] = attn
        return hook

    def generate_attribution(self, input_tensor, target_class=None):
        """
        Generate attribution map using attention rollout
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Get attention weights from all layers
        all_layer_attentions = []
        for name, attention in self.attention_maps.items():
            # Average over heads
            attn_weights = attention.mean(dim=1)  # [batch_size, seq_length, seq_length]
            all_layer_attentions.append(attn_weights)
        
        # Compute rollout attention
        cls_attention = compute_rollout_attention(all_layer_attentions)
        
        # Reshape to match image patches (14x14 for 224x224 image with patch size 16)
        attribution_map = cls_attention.reshape(-1, 14, 14)
        
        # Convert to numpy and normalize
        attribution_map = attribution_map.detach().cpu().numpy()
        attribution_map = attribution_map[0]  # Take first batch
        
        # Apply min-max normalization
        attribution_map = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min())
        
        # Resize to image size using bilinear interpolation
        attribution_map = cv2.resize(attribution_map, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        return attribution_map

    def compute_salience_scores(self, attribution_map, num_subsets=10):
        """
        Compute salience scores for K subsets of pixels sorted by importance.
        """
        flat_attribution = attribution_map.flatten()
        sorted_indices = np.argsort(flat_attribution)[::-1]
        
        pixels_per_subset = len(sorted_indices) // num_subsets
        subset_indices = [sorted_indices[i:i + pixels_per_subset] 
                         for i in range(0, len(sorted_indices), pixels_per_subset)]
        
        scores = []
        for indices in subset_indices:
            score = np.sum(flat_attribution[indices])
            scores.append(score)
            
        return subset_indices, np.array(scores)

    def perturb_image(self, input_image, subset_indices, strategy="mean"):
        """
        Perturb an input image by masking specific pixels.
        """
        device = input_image.device
        perturbed_image = input_image.clone()
        
        h, w = input_image.shape[2:]
        rows, cols = np.unravel_index(subset_indices, (h, w))
        
        if strategy == "mean":
            mask_value = self.imagenet_mean.to(device)
            mask_value = mask_value.view(3, 1).expand(3, len(rows))
        elif strategy == "black":
            mask_value = torch.zeros(3, len(rows), device=device)
        elif strategy == "random":
            mask_value = torch.randn(3, len(rows), device=device)
        else:
            raise ValueError(f"Unknown perturbation strategy: {strategy}")
            
        perturbed_image[0, :, rows, cols] = mask_value
        return perturbed_image

    def generate_visualization(self, input_tensor, target_class=None, use_thresholding=False):
        attribution_map = self.generate_attribution(input_tensor, target_class)
        
        if use_thresholding:
            attribution_map = attribution_map * 255
            attribution_map = attribution_map.astype(np.uint8)
            _, attribution_map = cv2.threshold(attribution_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            attribution_map = attribution_map / 255.0

        # Prepare image for visualization
        image = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        
        # Generate visualization
        vis = show_cam_on_image(image, attribution_map)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        
        return vis, attribution_map

def main():
    model_path = r'C:\Users\hp\Desktop\vs\ceng501-project\ViT-L-16_cifar10.pth'
    indices_path = r'C:\Users\hp\Desktop\vs\ceng501-project\cifar10_eval_indices.pkl'
    project_dir = r'C:\Users\hp\Desktop\vs\ceng501-project'
    
    # Create directories
    results_dir = os.path.join(project_dir, 'results')
    attribution_dir = os.path.join(results_dir, 'salience_maps', 'transformer_attribution')
    scores_dir = os.path.join(results_dir, 'salience_scores', 'transformer_attribution')
    perturb_dir = os.path.join(results_dir, 'perturbation_analysis', 'transformer_attribution')
    
    for directory in [attribution_dir, scores_dir, perturb_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_model('vit_large_patch16_224', pretrained=False, num_classes=10)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Use transform from the paper
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load dataset
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    with open(indices_path, 'rb') as f:
        eval_indices = pickle.load(f)
    dataset = torch.utils.data.Subset(dataset, eval_indices)
    
    attribution = TransformerAttribution(model)
    
    # Define mean and std for visualization
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    
    # Process images
    for i in range(10):
        image, label = dataset[i]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            pred = output.argmax(dim=1).item()
            
            # Print top predictions
            prob = torch.softmax(output, dim=1)[0]
            top5_prob, top5_idx = torch.topk(prob, 5)
            print(f"\nTop 5 predictions for image {i}:")
            for j, (p, idx) in enumerate(zip(top5_prob, top5_idx)):
                print(f"{j+1}. {class_names[idx]}: {p.item()*100:.2f}%")
        
        # Generate visualization
        vis, attribution_map = attribution.generate_visualization(image, target_class=None, use_thresholding=False)
        
        # Save basic visualization
        filename = f'attribution_img{i:02d}_{class_names[label]}_pred_{class_names[pred]}.png'
        save_path = os.path.join(attribution_dir, filename)
        cv2.imwrite(save_path, vis)
        
        # Compute salience scores and perturbation impact
        subset_indices, scores = attribution.compute_salience_scores(attribution_map)
        confidence_changes = []
        original_confidence = prob[pred].item()
        
        for indices in subset_indices:
            perturbed_image = attribution.perturb_image(image, indices, strategy="mean")
            with torch.no_grad():
                output = model(perturbed_image)
                confidence = torch.softmax(output, dim=1)[0, pred].item()
                confidence_change = original_confidence - confidence
                confidence_changes.append(confidence_change)
        
        # Save analysis results
        results_dict = {
            'image_index': i,
            'class_names': class_names,
            'original_class': class_names[label],
            'predicted_class': class_names[pred],
            'original_confidence': original_confidence,
            'salience_scores': scores.tolist(),
            'confidence_changes': confidence_changes,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        scores_file = os.path.join(scores_dir, f'attribution_scores_img{i:02d}.json')
        with open(scores_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f'Saved scores to: {scores_file}')
        
        # Save perturbation analysis visualization
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        img_display = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_display = (img_display * std) + mean  # Denormalize
        img_display = np.clip(img_display, 0, 1)
        plt.imshow(img_display)
        plt.title('Original Image')
        plt.axis('off')
        
        # Attribution map
        plt.subplot(1, 3, 2)
        plt.imshow(attribution_map, cmap='jet')
        plt.title('Attribution Map')
        plt.axis('off')
        
        # Perturbed image (using most important subset)
        plt.subplot(1, 3, 3)
        perturbed_image = attribution.perturb_image(
            image,
            subset_indices[0],  # Most important subset
            strategy="mean"
        )
        img_display = perturbed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_display = (img_display * std) + mean  # Denormalize
        img_display = np.clip(img_display, 0, 1)
        plt.imshow(img_display)
        plt.title('Perturbed Image\n(Most Important Subset)')
        plt.axis('off')
        
        plt.tight_layout()
        perturb_save_path = os.path.join(perturb_dir, f'perturbation_analysis_img{i:02d}.png')
        plt.savefig(perturb_save_path)
        plt.close()
        print(f'Saved perturbation analysis to: {perturb_save_path}')
        
        # Print analysis results
        print(f"\nAnalysis Results for image {i}:")
        print(f"Original class: {class_names[label]}")
        print(f"Predicted class: {class_names[pred]} (confidence: {original_confidence*100:.2f}%)")
        print("\nSalience Scores for each subset:")
        for j, score in enumerate(scores):
            print(f"Subset {j+1}: {score:.4f}")
        
        print("\nConfidence Changes after perturbation:")
        for j, change in enumerate(confidence_changes):
            print(f"Subset {j+1}: {change:.4f}")
        
        print(f'\nProcessed image {i+1}/10: {filename}')

if __name__ == '__main__':
    main()