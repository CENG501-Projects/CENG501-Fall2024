import timm
import torch
import logging

logger = logging.getLogger(__name__)

class ViTLarge:
    def __init__(self, num_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
    def load_model(self):
        """Load and return ViT-Large model"""
        logger.info("Loading ViT-Large model...")
        
        model = timm.create_model(
            'vit_large_patch16_224',
            pretrained=True,
            num_classes=self.num_classes
        )
        
        model.eval()
        model = model.to(self.device)
        
        logger.info("Successfully loaded ViT-Large model")
        logger.info(f"Model moved to {self.device}")
        
        return model 