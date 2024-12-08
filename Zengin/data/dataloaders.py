import os
import logging
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, config, dataset_name='cifar100'):
        """Initialize DatasetManager with config and dataset name"""
        
        if dataset_name not in config['datasets']:
            raise ValueError(f"Dataset {dataset_name} not found in config. Available datasets: {list(config['datasets'].keys())}")
            
        dataset_config = config['datasets'][dataset_name]
        self.dataset_name = dataset_config['name']
        
        project_root = Path(__file__).parent.parent
        self.raw_path = project_root / Path(dataset_config['raw_path'])
        self.processed_path = project_root / Path(dataset_config['processed_path'])
        
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        self.image_size = dataset_config['image_size']
        self.batch_size = dataset_config['batch_size']
        self.num_workers = dataset_config['num_workers']
        self.mean = dataset_config['mean']
        self.std = dataset_config['std']

        logger.info(f"Initialized DatasetManager for {self.dataset_name}")
        logger.info(f"Raw data path: {self.raw_path}")
        logger.info(f"Processed data path: {self.processed_path}")

    def get_transforms(self, train=True):
        """Define transformation pipeline"""
        if train:
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        return transform

    def get_dataset(self, train=True):
        """Download and return dataset
        
        Supported datasets:
        - CIFAR10: 10 classes, 50,000 training images, 10,000 test images
        - CIFAR100: 100 classes, 50,000 training images, 10,000 test images
        """
        logger.info(f"Loading {self.dataset_name} dataset...")
        
        if self.dataset_name.lower() == 'cifar100':
            dataset = torchvision.datasets.CIFAR100(
                root=str(self.raw_path),
                train=train,
                download=True,
                transform=self.get_transforms(train)
            )
            logger.info("Using CIFAR100 dataset with 100 classes")
        elif self.dataset_name.lower() == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(
                root=str(self.raw_path),
                train=train,
                download=True,
                transform=self.get_transforms(train)
            )
            logger.info("Using CIFAR10 dataset with 10 classes")
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported. Supported datasets: CIFAR10, CIFAR100")
            
        logger.info(f"Successfully loaded {self.dataset_name} dataset")
        return dataset

    def get_dataloader(self, train=True, shuffle=None):
        """Create and return DataLoader"""
        if shuffle is None:
            shuffle = train
            
        dataset = self.get_dataset(train)
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(
            f"Created dataloader with batch_size={self.batch_size}, "
            f"num_workers={self.num_workers}, shuffle={shuffle}"
        )
        return loader

def get_dataloaders(config_path="config.yaml", dataset_name='cifar100'):
    """Helper function to get train and test dataloaders"""
    
    project_root = Path(__file__).parent.parent
    config_path = project_root / config_path
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    manager = DatasetManager(config, dataset_name)
    
    train_loader = manager.get_dataloader(train=True)
    test_loader = manager.get_dataloader(train=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(dataset_name='cifar10')

