from models.backbone import Backbone
from models.att import SimplifiedATT
from models.mfr import SimplifiedMFR
import torch
from torchvision import transforms
from PIL import Image

# Initialize backbone, ATT, and MFR
backbone = Backbone(pretrained=False)
att = SimplifiedATT(num_clusters=16, embed_dim=512)
mfr = SimplifiedMFR(token_dim=512, out_channels=256, map_size=(7, 7))

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

# Pass through backbone, ATT, and MFR
features = backbone(input_tensor)
refined_tokens = att(features)
reconstructed_map = mfr(refined_tokens)
print(f"Reconstructed feature map shape: {reconstructed_map.shape}")

