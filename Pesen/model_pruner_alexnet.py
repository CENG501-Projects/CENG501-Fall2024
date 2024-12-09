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
            dataset_root = os.path.join(base_root, "imagenet\imagenet-object-localization-challenge")  # ImageNet-specific directory
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = ImageNetKaggle(root=dataset_root, split="train", transform=transform)
            test_dataset = ImageNetKaggle(root=dataset_root, split="val", transform=transform)

        else:
            raise ValueError("Unsupported dataset. Choose from 'mnist', 'cifar10', or 'imagenet'.")

        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True), DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def _extract_conv_weights(self):
        """Extract weights of all convolutional layers."""
        return [layer.weight.data.clone() for layer in self.model.modules() if isinstance(layer, nn.Conv2d)]
    
    def _recompute_fc_layer(self):
        """
        Recompute the input size of the first fully connected (FC) layer
        based on the output size of the pruned convolutional layers.
        """
        # Pass a dummy tensor through the model up to the FC layer
        dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)  # Example input
        features = self.model.features(dummy_input)  # Forward pass through convolutional layers
        new_flattened_size = features.view(-1).size(0)  # Calculate flattened size

        # Find the first Linear (fully connected) layer in the classifier
        for idx, layer in enumerate(self.model.classifier):
            if isinstance(layer, nn.Linear):
                # Replace the first Linear layer with updated input size
                self.model.classifier[idx] = nn.Linear(new_flattened_size, layer.out_features).to(self.device)
                logging.info(f"Replaced FC layer at index {idx} with input size: {new_flattened_size}")
                break


    def _update_model_weights(self, pruned_weights):
        """
        Update convolutional layers with pruned weights and adjust downstream layers for channel consistency.
        """
        index = 0
        prev_out_channels = None  # To store the number of output channels from the previous layer

        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                # Update current layer's weights
                layer.weight.data = pruned_weights[index]

                # Update bias if it exists
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[:pruned_weights[index].size(0)]

                # Update the next layer's input channels if necessary
                if prev_out_channels is not None and layer.weight.size(1) != prev_out_channels:
                    # Adjust the weights of the current layer to match the pruned output channels of the previous layer
                    layer.weight.data = layer.weight.data[:, :prev_out_channels, :, :]

                # Update for the next iteration
                prev_out_channels = pruned_weights[index].size(0)
                index += 1

        # Recompute FC layer dimensions
        self._recompute_fc_layer()




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
    model_name="alexnet",
    dataset_name="mnist",
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
