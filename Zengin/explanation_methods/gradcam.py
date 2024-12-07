import cv2  # OpenCV for image processing
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import numpy as np
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from timm import create_model
import json
import datetime

class GradCam:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.feature = None
        self.gradient = None
        self.handlers = []
        self.target = dict([*self.model.named_modules()])[target_layer]
        self._get_hook()
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _get_features_hook(self, module, input, output):
        self.feature = self.reshape_transform(output)

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = self.reshape_transform(output_grad[0])

    def _get_hook(self):
        self.handlers.append(self.target.register_forward_hook(self._get_features_hook))
        self.handlers.append(self.target.register_full_backward_hook(self._get_grads_hook))

    def reshape_transform(self, tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def generate_cam(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        target = output[0][target_class]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()
        weight = np.mean(gradient, axis=(1, 2))
        feature = self.feature[0].cpu().data.numpy()

        cam = feature * weight[:, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=0)
        cam = np.maximum(cam, 0)

        # Normalize the heatmap
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (224, 224))
        
        return cam

    def compute_salience_scores(self, salience_map, num_subsets=10):
        """
        Compute salience scores for K subsets of pixels sorted by importance.
        
        Args:
            salience_map (np.ndarray): The generated CAM
            num_subsets (int): Number of subsets to divide the salience map into
            
        Returns:
            tuple: (subset_indices, scores) where subset_indices is a list of pixel indices 
                  for each subset and scores is the sum of salience values for each subset
        """
        # Flatten and sort salience values
        flat_salience = salience_map.flatten()
        sorted_indices = np.argsort(flat_salience)[::-1]
        
        # Divide indices into K subsets
        pixels_per_subset = len(sorted_indices) // num_subsets
        subset_indices = [sorted_indices[i:i + pixels_per_subset] 
                         for i in range(0, len(sorted_indices), pixels_per_subset)]
        
        # Compute scores for each subset
        scores = []
        for indices in subset_indices:
            score = np.sum(flat_salience[indices])
            scores.append(score)
            
        return subset_indices, np.array(scores)

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
        
        if strategy == "mean":
            # Reshape mask value to match the dimensions of the target area
            mask_value = self.imagenet_mean.to(device)
            mask_value = mask_value.view(3, 1).expand(3, len(rows))
        elif strategy == "black":
            mask_value = torch.zeros(3, len(rows), device=device)
        elif strategy == "random":
            mask_value = torch.randn(3, len(rows), device=device)
        else:
            raise ValueError(f"Unknown perturbation strategy: {strategy}")
            
        # Apply mask
        perturbed_image[0, :, rows, cols] = mask_value
            
        return perturbed_image

    def analyze_perturbation_impact(self, input_image, target_class=None, num_subsets=10, 
                                  strategy="mean"):
        """
        Analyze the impact of perturbations on model predictions.
        
        Args:
            input_image (torch.Tensor): Input image tensor
            target_class (int, optional): Target class index
            num_subsets (int): Number of subsets for analysis
            strategy (str): Perturbation strategy
            
        Returns:
            dict: Analysis results including salience map, scores, and confidence changes
        """
        # Generate CAM
        cam = self.generate_cam(input_image, target_class)
        
        # Compute original confidence
        with torch.no_grad():
            original_output = self.model(input_image)
            if target_class is None:
                target_class = original_output.argmax(dim=1).item()
            original_confidence = F.softmax(original_output, dim=1)[0, target_class].item()
        
        # Compute salience scores
        subset_indices, scores = self.compute_salience_scores(cam, num_subsets)
        
        # Analyze perturbation impact
        confidence_changes = []
        for indices in subset_indices:
            # Perturb image
            perturbed_image = self.perturb_image(input_image, indices, strategy)
            
            # Compute new confidence
            with torch.no_grad():
                output = self.model(perturbed_image)
                confidence = F.softmax(output, dim=1)[0, target_class].item()
                confidence_change = original_confidence - confidence
                confidence_changes.append(confidence_change)
        
        return {
            'salience_map': cam,
            'salience_scores': scores,
            'confidence_changes': np.array(confidence_changes),
            'subset_indices': subset_indices
        }

    def __del__(self):
        for handler in self.handlers:
            handler.remove()

def load_model(model_path):
    model = create_model('vit_large_patch16_224', pretrained=False, num_classes=10)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_dataset(indices_path):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    with open(indices_path, 'rb') as f:
        eval_indices = pickle.load(f)
    
    return torch.utils.data.Subset(dataset, eval_indices)

def visualize_gradcam(image_tensor, cam, save_path):
    image = image_tensor.permute(1, 2, 0).numpy()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title('Grad-CAM')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    model_path = r'C:\Users\hp\Desktop\vs\ceng501-project\ViT-L-16_cifar10.pth'
    indices_path = r'C:\Users\hp\Desktop\vs\ceng501-project\cifar10_eval_indices.pkl'
    
    # Fix directory paths to use the main project folder
    project_dir = r'C:\Users\hp\Desktop\vs\ceng501-project'  # Direct path to main project folder
    results_dir = os.path.join(project_dir, 'results')
    
    # Create main results directories
    gradcam_dir = os.path.join(results_dir, 'salience_maps', 'cifar10', 'gradcam')
    scores_dir = os.path.join(results_dir, 'salience_scores', 'gradcam')
    perturb_dir = os.path.join(results_dir, 'perturbation_analysis', 'gradcam')
    
    # Create all directories
    for directory in [gradcam_dir, scores_dir, perturb_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print(f"Saving salience scores to: {scores_dir}")  # Debug print
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(model_path)
    model = model.to(device)
    
    gradcam = GradCam(model, target_layer='blocks.23.norm1') 
    
    dataset = load_dataset(indices_path)
    
    # Define ImageNet mean and std for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Test case for new functionality
    image, label = dataset[0]
    image = image.unsqueeze(0).to(device)
    
    # Analyze perturbation impact
    results = gradcam.analyze_perturbation_impact(
        image, 
        target_class=None,  # Use predicted class
        num_subsets=10, 
        strategy="mean"
    )
    
    # Print analysis results
    print("\nAnalysis Results:")
    print(f"Original class: {class_names[label]}")
    print("\nSalience Scores for each subset:")
    for i, score in enumerate(results['salience_scores']):
        print(f"Subset {i+1}: {score:.4f}")
    
    print("\nConfidence Changes after perturbation:")
    for i, change in enumerate(results['confidence_changes']):
        print(f"Subset {i+1}: {change:.4f}")
    
    # Visualize original and perturbed images
    # Perturb image using the most important subset
    perturbed_image = gradcam.perturb_image(
        image,
        results['subset_indices'][0],  # Most important subset
        strategy="mean"
    )
    
    # Save visualization
    output_dir = os.path.join(project_dir, 'results', 'perturbation_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    img_display = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)
    plt.imshow(img_display)
    plt.title('Original Image')
    plt.axis('off')
    
    # Salience map
    plt.subplot(1, 3, 2)
    plt.imshow(results['salience_map'], cmap='jet')
    plt.title('Salience Map')
    plt.axis('off')
    
    # Perturbed image
    plt.subplot(1, 3, 3)
    img_display = perturbed_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)
    plt.imshow(img_display)
    plt.title('Perturbed Image\n(Most Important Subset)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'perturbation_analysis.png'))
    plt.close()
    
    # Modify the visualization loop
    for i in range(10):
        image, label = dataset[i]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(image).argmax(dim=1).item()
        
        # Generate Grad-CAM and analyze
        results = gradcam.analyze_perturbation_impact(
            image, 
            target_class=None,
            num_subsets=10, 
            strategy="mean"
        )
        
        # Save visualization
        filename = f'gradcam_img{i:02d}_{class_names[label]}_pred_{class_names[pred]}.png'
        save_path = os.path.join(gradcam_dir, filename)
        visualize_gradcam(image.squeeze(0).cpu(), results['salience_map'], save_path)
        
        # Save scores with more detailed information
        results_dict = {
            'image_index': i,
            'class_names': class_names,
            'original_class': class_names[label],
            'predicted_class': class_names[pred],
            'salience_scores': results['salience_scores'].tolist(),
            'confidence_changes': results['confidence_changes'].tolist(),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        scores_file = os.path.join(scores_dir, f'gradcam_scores_img{i:02d}.json')
        try:
            with open(scores_file, 'w') as f:
                json.dump(results_dict, f, indent=4)
            print(f'Saved scores to: {scores_file}')
        except Exception as e:
            print(f'Error saving scores: {e}')
        
        print(f'Processed image {i+1}/10: {filename}')
        
        # Save perturbation analysis visualization
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        img_display = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)
        plt.imshow(img_display)
        plt.title('Original Image')
        plt.axis('off')
        
        # Salience map
        plt.subplot(1, 3, 2)
        plt.imshow(results['salience_map'], cmap='jet')
        plt.title('Salience Map')
        plt.axis('off')
        
        # Perturbed image (using most important subset)
        plt.subplot(1, 3, 3)
        perturbed_image = gradcam.perturb_image(
            image,
            results['subset_indices'][0],  # Most important subset
            strategy="mean"
        )
        img_display = perturbed_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)
        plt.imshow(img_display)
        plt.title('Perturbed Image\n(Most Important Subset)')
        plt.axis('off')
        
        plt.tight_layout()
        perturb_save_path = os.path.join(perturb_dir, f'perturbation_analysis_img{i:02d}.png')
        plt.savefig(perturb_save_path)
        plt.close()
        print(f'Saved perturbation analysis to: {perturb_save_path}')

if __name__ == '__main__':
    main()
