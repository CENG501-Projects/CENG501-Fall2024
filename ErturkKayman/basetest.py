from models.mono3d_detection import MonoATT
import torch
from torchvision import transforms
from PIL import Image

# Initialize MonoATT model
model = MonoATT()

# Test image
test_image_path = "data/KITTIDataset/training/image_2/000000.png"
test_image = Image.open(test_image_path).convert("RGB")

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for faster testing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(test_image).unsqueeze(0)  # Add batch dimension

# Forward pass through MonoATT
output = model(input_tensor)
print(f"MonoATT output shape: {output.shape}")

