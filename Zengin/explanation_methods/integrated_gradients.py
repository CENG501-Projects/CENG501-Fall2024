import torch
from captum.attr import IntegratedGradients
import numpy as np
import matplotlib.pyplot as plt
from timm import create_model
import os
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import pickle

class IntegratedGradientsExplainer:
    def __init__(self, model):
        """
        Initialize Integrated Gradients for Vision Transformer.
        Args:
            model: The fine-tuned Vision Transformer model.
        """
        self.model = model
        self.model.eval()
        self.ig = IntegratedGradients(self.model)

    def generate_attributions(self, input_tensor, target_class):
        """
        Generate Integrated Gradients attributions.
        Args:
            input_tensor: The input image tensor (1, 3, 224, 224).
            target_class: The target class for explanation.
        Returns:
            Attribution map as a numpy array.
        """
        # Baseline: Black image (all zeros)
        baseline = torch.zeros_like(input_tensor).to(input_tensor.device)

    
        attributions = self.ig.attribute(
            inputs=input_tensor,
            baselines=baseline,
            target=target_class,
            n_steps=50
        )
        
        
        aggregated_attributions = attributions.sum(dim=1).squeeze()
        
        
        normalized_attributions = (aggregated_attributions - aggregated_attributions.min()) / (
            aggregated_attributions.max() - aggregated_attributions.min())
        
        return normalized_attributions.detach().cpu().numpy()

    def visualize_attributions(self, image_tensor, attributions, save_path):
        """
        Visualize attribution maps.
        Args:
            image_tensor: Original input image tensor
            attributions: Attribution map
            save_path: Path to save visualization
        """
       
        image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

       
        plt.figure(figsize=(10, 3))
        
        plt.subplot(131)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(132)
        plt.imshow(attributions, cmap="hot")
        plt.title("Attribution Map")
        plt.axis("off")
        
        plt.subplot(133)
        plt.imshow(image)
        plt.imshow(attributions, cmap="hot", alpha=0.5)
        plt.title("Overlay")
        plt.axis("off")
        
        plt.savefig(save_path)
        plt.close()

def main():
   
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'ViT-L-16_cifar10.pth')
    indices_path = os.path.join(base_dir, 'cifar10_eval_indices.pkl')
    output_dir = os.path.join(base_dir, 'results', 'salience_maps', 'cifar10', 'integrated_gradients')
    os.makedirs(output_dir, exist_ok=True)

    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    model = create_model('vit_large_patch16_224', pretrained=False, num_classes=10)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    explainer = IntegratedGradientsExplainer(model)


    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cifar10_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    with open(indices_path, 'rb') as f:
        eval_indices = pickle.load(f)
    eval_dataset = Subset(cifar10_test, eval_indices)


    for i in range(10):
        image, label = eval_dataset[i]
        image = image.unsqueeze(0).to(device)
        
       
        with torch.no_grad():
            pred = model(image).argmax(dim=1).item()

        
        attributions = explainer.generate_attributions(image, target_class=pred)

       
        filename = f'ig_img{i:02d}_{class_names[label]}_pred_{class_names[pred]}.png'
        save_path = os.path.join(output_dir, filename)
        explainer.visualize_attributions(image, attributions, save_path)

        print(f'Processed image {i+1}/10: {filename}')

if __name__ == "__main__":
    main()

