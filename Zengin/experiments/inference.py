import os
import logging
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import yaml
import sys


project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)


from models.vit_b import ViTBase
from models.vit_l import ViTLarge
from models.deit_b import DeiTBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        
        project_root = Path(__file__).parent.parent
        self.results_dir = project_root / 'results' / 'predictions'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_models(self, num_classes):
        models = {
            'vit_b_16': ViTBase(num_classes).load_model(),
            'vit_l_16': ViTLarge(num_classes).load_model(),
            'deit_b': DeiTBase(num_classes).load_model()
        }
        return models

    def process_image(self, image_path):
        image = Image.open(image_path)
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)

    def run_inference(self, dataset_path, num_classes):
        logger.info(f"Running inference on {dataset_path}")
        
        
        logger.info("Loading models...")
        models = self.load_models(num_classes)
        logger.info(f"Loaded {len(models)} models successfully")
        
        
        model_results = {
            model_name: {
                'image_path': [],
                'true_label': [],
                'prediction': [],
                'confidence': []
            }
            for model_name in models.keys()
        }
        
        
        test_dir = Path(dataset_path) / 'test'
        total_images = sum(len(list(class_dir.glob('*.png'))) for class_dir in test_dir.iterdir())
        logger.info(f"Found {total_images} images to process")
        
        
        with tqdm(total=total_images, desc="Processing images") as pbar:
            for class_dir in sorted(test_dir.iterdir()):
                true_label = int(class_dir.name)
                
                for image_path in sorted(class_dir.glob('*.png')):
                    image_tensor = self.process_image(image_path)
                    
                    
                    for model_name, model in models.items():
                        with torch.no_grad():
                            outputs = model(image_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            
                            pred_label = torch.argmax(probabilities, dim=1).item()
                            confidence = probabilities[0][pred_label].item()
                            
                            
                            model_results[model_name]['image_path'].append(str(image_path))
                            model_results[model_name]['true_label'].append(true_label)
                            model_results[model_name]['prediction'].append(pred_label)
                            model_results[model_name]['confidence'].append(confidence)
                    
                    
                    pbar.update(1)
        
        
        return {
            model_name: pd.DataFrame(results)
            for model_name, results in model_results.items()
        }

def run_all_inference(config_path=r"ceng-project\config.yaml"):
    """Run inference on all processed datasets"""
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    inference = ModelInference(config)
    
   
    project_root = Path(__file__).parent.parent
    
    
    logger.info("Processing CIFAR10...")
    cifar10_path = project_root / 'data' / 'processed' / 'cifar10_224'
    if not (cifar10_path / 'test').exists():
        raise FileNotFoundError(f"Test directory not found at {cifar10_path / 'test'}")
    cifar10_results = inference.run_inference(cifar10_path, num_classes=10)
    
    
    for model_name, results_df in cifar10_results.items():
        save_path = inference.results_dir / f'cifar10_{model_name}_predictions.csv'
        results_df.to_csv(save_path, index=False)
        logger.info(f"Saved {model_name} CIFAR10 results to: {save_path}")
    
    
    logger.info("Processing CIFAR100...")
    cifar100_path = project_root / 'data' / 'processed' / 'cifar100_224'
    if not (cifar100_path / 'test').exists():
        raise FileNotFoundError(f"Test directory not found at {cifar100_path / 'test'}")
    cifar100_results = inference.run_inference(cifar100_path, num_classes=100)
    
    
    for model_name, results_df in cifar100_results.items():
        save_path = inference.results_dir / f'cifar100_{model_name}_predictions.csv'
        results_df.to_csv(save_path, index=False)
        logger.info(f"Saved {model_name} CIFAR100 results to: {save_path}")
    
    logger.info("Inference complete. Results saved in results/predictions/")

if __name__ == "__main__":
    run_all_inference()