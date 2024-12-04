import os
import logging
import torch
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path
import yaml
from tqdm import tqdm
from dataloaders import DatasetManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(self, config, dataset_name):
        """Initialize processor with config and dataset name"""
        self.manager = DatasetManager(config, dataset_name)
        self.processed_path = self.manager.processed_path
        self.dataset_name = dataset_name
        
        
        self.test_dir = self.processed_path / 'test'
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def save_processed_image(self, img_tensor, label, idx):
        """Save processed tensor as image with its label"""
        
        img_array = img_tensor.permute(1, 2, 0).numpy()
       
        img_array = (img_array * np.array(self.manager.std) + np.array(self.manager.mean)) * 255
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        
        class_dir = self.test_dir / str(label)
        class_dir.mkdir(exist_ok=True)
        
        img_path = class_dir / f"{idx:05d}.png"
        img.save(img_path)

    def process_test_dataset(self):
        """Process and save test dataset only"""
        logger.info(f"Processing {self.dataset_name} test dataset...")
        
       
        test_dataset = self.manager.get_dataset(train=False)
        logger.info(f"Processing {len(test_dataset)} test images...")
        for idx, (img, label) in enumerate(tqdm(test_dataset)):
            self.save_processed_image(img, label, idx)
            
        logger.info(f"Finished processing {self.dataset_name} test dataset")
        logger.info(f"Processed test data saved to {self.test_dir}")

def process_test_datasets(config_path="config.yaml"):
    """Process test sets for all datasets defined in config"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / config_path
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    for dataset_name in config['datasets'].keys():
        processor = DatasetProcessor(config, dataset_name)
        processor.process_test_dataset()

if __name__ == "__main__":
    process_test_datasets() 
