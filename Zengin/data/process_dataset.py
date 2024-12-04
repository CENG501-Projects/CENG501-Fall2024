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
        
        # Create train and test directories
        self.train_dir = self.processed_path / 'train'
        self.test_dir = self.processed_path / 'test'
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def save_processed_image(self, img_tensor, label, idx, is_train):
        """Save processed tensor as image with its label"""
        
        img_array = img_tensor.permute(1, 2, 0).numpy()
        
        img_array = (img_array * np.array(self.manager.std) + np.array(self.manager.mean)) * 255
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        
        save_dir = self.train_dir if is_train else self.test_dir
        class_dir = save_dir / str(label)
        class_dir.mkdir(exist_ok=True)
        
        img_path = class_dir / f"{idx:05d}.png"
        img.save(img_path)

    def process_dataset(self):
        """Process and save both train and test datasets"""
        logger.info(f"Processing {self.dataset_name} dataset...")
        
       
        train_dataset = self.manager.get_dataset(train=True)
        logger.info("Processing training data...")
        for idx, (img, label) in enumerate(tqdm(train_dataset)):
            self.save_processed_image(img, label, idx, is_train=True)
            
        
        test_dataset = self.manager.get_dataset(train=False)
        logger.info("Processing test data...")
        for idx, (img, label) in enumerate(tqdm(test_dataset)):
            self.save_processed_image(img, label, idx, is_train=False)
            
        logger.info(f"Finished processing {self.dataset_name} dataset")
        logger.info(f"Processed data saved to {self.processed_path}")

def process_all_datasets(config_path="config.yaml"):
    """Process all datasets defined in config"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / config_path
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    for dataset_name in config['datasets'].keys():
        processor = DatasetProcessor(config, dataset_name)
        processor.process_dataset()

if __name__ == "__main__":
    process_all_datasets() 
