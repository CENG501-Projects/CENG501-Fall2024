import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from filter_pruner import FilterPruner
from afie_and_pruning_ratio import LayerPruningAnalyzer
from imagenet_kaggle import ImageNetKaggle
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelPruner:
    def __init__(self, model_name, dataset_name, overall_pruning_ratio=0.5, lambda_min=0.1, batch_size=64, device='cpu'):
        """Initialize the ModelPruner with model and dataset details."""
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.overall_pruning_ratio = overall_pruning_ratio
        self.lambda_min = lambda_min
        self.batch_size = batch_size
        self.device = device

        # Load the model and dataset
        self.model = self._load_model(model_name).to(self.device)
        self.model.eval()
        self.train_loader, self.test_loader = self._load_dataset(dataset_name)

    def _load_model(self, model_name):
        """Load a pre-trained model."""
        if model_name == "alexnet":
            return models.alexnet(pretrained=True)
        elif model_name == "vgg16":
            return models.vgg16(pretrained=True)
        elif model_name == "resnet50":
            return models.resnet50(pretrained=True)
        else:
            raise ValueError("Unsupported model. Choose from 'alexnet', 'vgg16', or 'resnet50'.")

    def _load_dataset(self, dataset_name):
        """
        Load the dataset with preprocessing.
        """
        base_root = "./data"  # Base directory for all datasets
        if dataset_name == "mnist":
            dataset_root = os.path.join(base_root, "mnist")  # MNIST-specific directory
            transform = transforms.Compose([
                transforms.Resize(224),  # Resize to 224x224 for AlexNet
                transforms.Lambda(lambda x: x.convert("RGB")),  # Convert grayscale to RGB
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels
            ])
            train_dataset = datasets.MNIST(root=dataset_root, train=True, transform=transform, download=True)
            test_dataset = datasets.MNIST(root=dataset_root, train=False, transform=transform, download=True)

        elif dataset_name == "cifar10":
            dataset_root = os.path.join(base_root, "cifar10")  # CIFAR-10-specific directory
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(root=dataset_root, train=True, transform=transform, download=True)
            test_dataset = datasets.CIFAR10(root=dataset_root, train=False, transform=transform, download=True)

        elif dataset_name == "imagenet":
            dataset_root = os.path.join(base_root, "imagenet\\imagenet-object-localization-challenge")  # ImageNet-specific directory
            transform = transforms.Compose([
                transforms.Resize(256),  # Resize the smaller dimension to 256
                transforms.CenterCrop(224),  # Crop the image to 224x224
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Standard normalization
            ])
            train_dataset = ImageNetKaggle(root=dataset_root, split="train", transform=transform)
            test_dataset = ImageNetKaggle(root=dataset_root, split="val", transform=transform)

        else:
            raise ValueError("Unsupported dataset. Choose from 'mnist', 'cifar10', or 'imagenet'.")

        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True), DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def _extract_conv_weights(self):
        """Extract weights of all convolutional layers."""
        return [layer.weight.data.clone() for layer in self.model.modules() if isinstance(layer, nn.Conv2d)]

    def _update_batch_norm_layer(self, bn_layer, num_features):
        """
        Update a BatchNorm2d layer to match the new number of features.
        """
        current_features = bn_layer.num_features
        logging.info(f"Updating BatchNorm2d: Expected num_features={num_features}, Current BN size={current_features}")
        
        if num_features != current_features:
            logging.warning(f"Mismatch detected. Adjusting BatchNorm2d from {current_features} to {num_features}.")

        new_bn_layer = nn.BatchNorm2d(num_features).to(self.device)
        with torch.no_grad():
            min_features = min(current_features, num_features)
            new_bn_layer.weight[:min_features] = bn_layer.weight[:min_features]
            new_bn_layer.bias[:min_features] = bn_layer.bias[:min_features]
            new_bn_layer.running_mean[:min_features] = bn_layer.running_mean[:min_features]
            new_bn_layer.running_var[:min_features] = bn_layer.running_var[:min_features]
        
        logging.info(f"Updated BatchNorm2d layer: {new_bn_layer}")
        return new_bn_layer


    def _recompute_fc_layer(self):
        """Recompute the fully connected layer and update batch normalization layers."""
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        if self.model_name == "resnet50":
            # Update conv1 and bn1
            features = self.model.conv1(dummy_input)
            num_features = self.model.conv1.weight.data.size(0)  # Ensure consistency with actual weights
            
            logging.info(f"Recomputing conv1: num_features={num_features}, conv1_weights.size={self.model.conv1.weight.data.size()}")
            
            if features.size(1) != num_features:
                raise RuntimeError(
                    f"Mismatch after conv1: Features have {features.size(1)} channels but conv1 weights have {num_features}."
                )
            
            self.model.bn1 = self._update_batch_norm_layer(self.model.bn1, num_features)
            features = self.model.bn1(features)
            features = self.model.relu(features)
            features = self.model.maxpool(features)

            # Propagate through layers
            for layer in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]:
                for block in layer:
                    # Update batch norms in the block
                    if hasattr(block, "bn1"):
                        block.bn1 = self._update_batch_norm_layer(block.bn1, block.conv1.out_channels)
                    if hasattr(block, "bn2"):
                        block.bn2 = self._update_batch_norm_layer(block.bn2, block.conv2.out_channels)
                    if hasattr(block, "bn3"):
                        block.bn3 = self._update_batch_norm_layer(block.bn3, block.conv3.out_channels)

                    # Validate block processing
                    features = block(features)
                    logging.info(f"Processed block: {block}")

        else:
            # For AlexNet/VGG
            features = self.model.features(dummy_input)

        # Update FC layer
        num_features = features.view(features.size(0), -1).size(1)
        old_fc = self.model.fc
        self.model.fc = nn.Linear(num_features, old_fc.out_features).to(self.device)


    def _recompute_bn_stats(self, bn_layer, num_features):
        """
        Recompute the running statistics (mean, var) for BatchNorm2d layers after pruning.

        Args:
            bn_layer (nn.BatchNorm2d): The batch normalization layer to recompute stats for.
            num_features (int): The new number of output features (channels).

        Returns:
            nn.BatchNorm2d: The updated batch normalization layer.
        """
        with torch.no_grad():
            # Reset the running statistics to zero
            bn_layer.running_mean.data.zero_()
            bn_layer.running_var.data.fill_(1.0)

            # Update the batch normalization layer to reflect the new number of channels
            bn_layer.num_features = num_features
            bn_layer.weight.data = torch.ones(num_features)  # Reset weight to 1 for all channels
            bn_layer.bias.data.zero_()  # Reset bias to 0 for all channels

        return bn_layer

    def _update_model_weights(self, pruned_weights):
        """Update the model's convolutional layers with pruned weights."""
        index = 0
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                pruned_channels = pruned_weights[index].size(0)
                original_channels = layer.weight.data.size(0)
                
                if pruned_channels < original_channels:
                    logging.warning(
                        f"Pruned channels ({pruned_channels}) are fewer than expected ({original_channels})."
                        f" Adjusting to maintain compatibility."
                    )
                    # Add dummy channels or skip pruning for this layer
                    padded_weights = torch.cat([
                        pruned_weights[index],
                        torch.zeros(original_channels - pruned_channels, *pruned_weights[index].size()[1:]).to(self.device)
                    ])
                    layer.weight.data = padded_weights
                else:
                    layer.weight.data = pruned_weights[index]
                index += 1



    def compute_afie_scores(self, conv_weights):
        """Compute AFIE scores for convolutional layers using LayerPruningAnalyzer."""
        afie_scores = []
        for weight in conv_weights:
            weight_matrix = weight.view(weight.size(0), -1)  # Reshape to 2D
            analyzer = LayerPruningAnalyzer(weight_matrix)
            afie = analyzer.compute_afie_and_pruning_ratio()[0]  # Only need AFIE score
            afie_scores.append(afie)
        return afie_scores

    def prune_and_fine_tune(self, epochs=5, lr=0.001):
        """Prune filters in the model and fine-tune."""
        # Step 1: Extract convolutional layer weights
        conv_weights = self._extract_conv_weights()

        # Step 2: Compute AFIE scores for each layer using LayerPruningAnalyzer
        afie_scores = self.compute_afie_scores(conv_weights)

        # Step 3: Initialize FilterPruner
        num_filters_per_layer = [weight.size(0) for weight in conv_weights]
        pruner = FilterPruner(afie_scores, num_filters_per_layer)
        pruning_ratios = pruner.compute_pruning_ratios(self.overall_pruning_ratio, self.lambda_min)
        logging.info(f"Pruning Ratios: {pruning_ratios}")

        # Step 4: Prune filters
        pruned_weights = pruner.prune_filters(conv_weights, pruning_ratios)

        # Step 5: Update the model with pruned weights
        self._update_model_weights(pruned_weights)

        # Step 6: Fine-tune the pruned model
        self.fine_tune(epochs, lr)

    def fine_tune(self, epochs, lr):
        """Fine-tune the pruned model with early stopping."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        best_accuracy, patience, patience_counter = 0, 5, 0
        for epoch in range(epochs):
            self.model.train()
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

            val_accuracy = self.evaluate_model()
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info("Early stopping triggered.")
                    break

    def evaluate_model(self):
        """Evaluate the model and return accuracy."""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        logging.info(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy


"""
pruner = ModelPruner(
    model_name="resnet50",
    dataset_name="mnist",
    overall_pruning_ratio=0.5,
    batch_size=64,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
train_loader, test_loader = pruner._load_dataset("mnist")
print(len(train_loader.dataset), len(test_loader.dataset))
"""
pruner = ModelPruner(
    model_name="resnet50",
    dataset_name="imagenet",
    overall_pruning_ratio=0.5,
    batch_size=64,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Initialize results storage
results = []

# Train and prune the model, logging results at 1, 10, and 20 epochs
for epoch_limit in [1, 50, 150]:
    logging.info(f"Running pruning and fine-tuning for {epoch_limit} epochs.")
    pruner.prune_and_fine_tune(epochs=epoch_limit, lr=0.001)
    
    # Compute final accuracy
    final_accuracy = pruner.evaluate_model()
    logging.info(f"Epochs: {epoch_limit}, Final Accuracy: {final_accuracy:.2f}%")
    
    # Extract and compute layer-wise AFIE scores after pruning
    conv_weights = pruner._extract_conv_weights()  # Extract convolutional layer weights
    afie_scores = pruner.compute_afie_scores(conv_weights)  # Compute AFIE scores
    
    # Log results
    logging.info(f"AFIE Scores (Layer-wise): {afie_scores}")
    results.append({
        "epochs": epoch_limit,
        "accuracy": final_accuracy,
        "afie_scores": afie_scores
    })

# Display final results
for result in results:
    print(f"Epochs: {result['epochs']}, Accuracy: {result['accuracy']:.2f}%, AFIE: {result['afie_scores']}")
