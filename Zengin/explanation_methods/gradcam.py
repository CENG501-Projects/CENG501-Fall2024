import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import numpy as np
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from timm import create_model

class VitGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        
        target = dict([*self.model.named_modules()])[self.target_layer]
        target.register_forward_hook(self.save_activation)
        target.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, target_class=None):
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        
        self.model.zero_grad()
        
        
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        
        gradients = self.gradients[0] 
        activations = self.activations[0]  
        
        
        gradients = gradients[1:] 
        activations = activations[1:]  
        
        
        weights = torch.mean(gradients, dim=1)
        
        
        cam = weights * torch.norm(activations, dim=1)
        
        
        cam = cam.reshape(14, 14)  
        cam = F.relu(cam)
        
        
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )
        
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.detach().cpu().numpy()

def load_model(model_path):
    model = create_model('vit_base_patch16_224', pretrained=False, num_classes=10)
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
    
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    model_path = os.path.join(base_dir, 'ViT-B-16_cifar10.pth')
    indices_path = os.path.join(base_dir, 'cifar10_eval_indices.pkl')
    output_dir = os.path.join(base_dir, 'results', 'salience_maps', 'cifar10', 'gradcam')
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    model = load_model(model_path)
    model = model.to(device)
    
    
    gradcam = VitGradCAM(model, target_layer='blocks.11.norm1')
    
    
    dataset = load_dataset(indices_path)
    
    
    for i in range(10):
        image, label = dataset[i]
        image = image.unsqueeze(0).to(device)
        
        
        with torch.no_grad():
            pred = model(image).argmax(dim=1).item()
        
        
        cam = gradcam.generate_cam(image)
        
        
        filename = f'gradcam_img{i:02d}_{class_names[label]}_pred_{class_names[pred]}.png'
        save_path = os.path.join(output_dir, filename)
        
        
        visualize_gradcam(image.squeeze(0).cpu(), cam, save_path)
        
        print(f'Processed image {i+1}/10: {filename}')

if __name__ == '__main__':
    main()
