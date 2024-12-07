import torch
import numpy as np
import matplotlib.pyplot as plt
from timm import create_model
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
import pickle
import gc
import os
from captum.attr import IntegratedGradients

class IGExplainer:
    def __init__(self, model):
        self.model = model.eval()
        self.ig = IntegratedGradients(self.model)
        
    def generate_attributions(self, input_tensor, target_class=None):
        
        baseline = torch.rand_like(input_tensor).to(input_tensor.device) * 0.5
        
       
        attributions = self.ig.attribute(
            input_tensor,
            target=target_class,
            baselines=baseline,
            n_steps=50,
            return_convergence_delta=False
        )
        
      
        attributions = attributions.sum(dim=1).squeeze()
        attributions = torch.relu(attributions)
        if attributions.max() > 0:
            attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())
        
       
        print("Attributions stats - min:", attributions.min().item(), 
              "max:", attributions.max().item(), "var:", attributions.var().item())
        
        return attributions.cpu().detach().numpy()
    
    def visualize_attributions(self, image_tensor, attributions, save_path):
       
        image = image_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(132)
        plt.imshow(attributions, cmap="hot")
        plt.title("Attribution Map")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
        
        plt.subplot(133)
        plt.imshow(image)
        plt.imshow(attributions, cmap="hot", alpha=0.7)  
        plt.title("Overlay")
        plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


output_dir = r'C:\Users\hp\Desktop\vs\ceng501-project\results\salience_maps\integrated_gradients'
os.makedirs(output_dir, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load ViT-L model
print("Loading ViT-L model...")
model_path = r'C:\Users\hp\Desktop\vs\ceng501-project\ViT-L-16_cifar10.pth'
model = create_model('vit_large_patch16_224', pretrained=False, num_classes=10)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()


explainer = IGExplainer(model)


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cifar10_test = CIFAR10(root='./data', train=False, download=True, transform=transform)


indices_path = r'C:\Users\hp\Desktop\vs\ceng501-project\cifar10_eval_indices.pkl'
with open(indices_path, 'rb') as f:
    eval_indices = pickle.load(f)
eval_dataset = Subset(cifar10_test, eval_indices)


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Process 10 images
for i in range(10):
    image, label = eval_dataset[i]
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(image).argmax(dim=1).item()
    
    print(f"Processing image {i+1}/10...")
    print(f"Predicted: {class_names[pred]}, Actual: {class_names[label]}")
    
    attributions = explainer.generate_attributions(image, target_class=pred)
    
    filename = f'ig_img{i:02d}_{class_names[label]}_pred_{class_names[pred]}.png'
    save_path = os.path.join(output_dir, filename)
    explainer.visualize_attributions(image, attributions, save_path)
    
    print(f'Image {i+1}/10 processed: {filename}')
    
  
    torch.cuda.empty_cache()
    gc.collect()

print("\nProcessing complete! Images saved to:", output_dir)
