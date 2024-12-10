from models.backbone import Backbone
import torch
from torchvision import transforms
from PIL import Image

# Initialize backbone
backbone = Backbone(pretrained=False)

# Test image (use a random image from KITTI)
test_image_path = "data/KITTIDataset/training/image_2/000000.png"
test_image = Image.open(test_image_path).convert("RGB")

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for faster testing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(test_image).unsqueeze(0)  # Add batch dimension

# Pass through backbone
features = backbone(input_tensor)
print(f"Extracted feature shape: {features.shape}")

