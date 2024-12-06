import cv2 
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import numpy as np
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from timm import create_model

class GradCam:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.feature = None
        self.gradient = None
        self.handlers = []
        self.target = dict([*self.model.named_modules()])[target_layer]
        self._get_hook()

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
    
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'results', 'salience_maps', 'cifar10', 'gradcam')
    
    os.makedirs(output_dir, exist_ok=True)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(model_path)
    model = model.to(device)
    
    gradcam = GradCam(model, target_layer='blocks.23.norm1') 
    
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
