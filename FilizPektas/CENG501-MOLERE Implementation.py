# -*- coding: utf-8 -*-
"""Untitled20.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VkXkz0yQpfp2JPwXCGLOaKbUMw3c9yyS
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
import time

# Define the WideResNet architecture
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )
        self.dropout_rate = dropout_rate

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=100, dropout_rate=0.3):
        super(WideResNet, self).__init__()
        self.in_channels = 16
        assert (depth - 4) % 6 == 0, 'Depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(16 * k, n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(32 * k, n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(64 * k, n, stride=2, dropout_rate=dropout_rate)
        self.bn = nn.BatchNorm2d(64 * k)
        self.linear = nn.Linear(64 * k, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride, dropout_rate))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Training and evaluation functions
def evaluate(model, data_loader, device, num_evaluations=5):
    model.eval()
    correct = 0
    total = 0
    prediction_variances = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            # Perform multiple forward passes for variance estimation
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = []
            for _ in range(num_evaluations):
                outputs = model(inputs)
                predictions.append(outputs.softmax(dim=1).cpu().numpy())  # Softmax for probabilities

            predictions = np.array(predictions)  # Shape: (num_evaluations, batch_size, num_classes)
            mean_predictions = predictions.mean(axis=0)  # Mean over evaluations
            variance = predictions.var(axis=0)  # Variance over evaluations

            # Append variances for each input
            prediction_variances.extend(variance.max(axis=1))  # Max variance across classes for each input

            # Calculate accuracy
            _, predicted = torch.tensor(mean_predictions).max(1)  # Use mean predictions for accuracy
            total += labels.size(0)
            correct += predicted.eq(labels.cpu()).sum().item()
    return 100. * correct / total

def enable_dropout(model):
    """
    Enable dropout layers during evaluation for Monte Carlo Dropout.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()  # Force dropout layers to train mode

def estimate_variance(model, inputs, num_evaluations=5):
    """
    Estimate variance of predictions using Monte Carlo Dropout.
    Args:
        model: PyTorch model with dropout layers.
        inputs: Input data (torch.Tensor).
        num_evaluations: Number of stochastic forward passes.

    Returns:
        mean_predictions: Mean of predictions over multiple evaluations.
        variance: Variance of predictions over multiple evaluations.
    """
    model.eval()  # Ensure other layers are in evaluation mode
    enable_dropout(model)  # Enable dropout during evaluation

    predictions = []
    with torch.no_grad():
        for _ in range(num_evaluations):
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())

    predictions = np.array(predictions)
    mean_predictions = predictions.mean(axis=0)
    variance = predictions.var(axis=0)

    return mean_predictions, variance

def train(model, train_loader, val_loader, device, epochs=100):
  model = model.to()
  optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

  # Warm-Up Parameters
  warmup_epochs = 25
  initial_lr = 1e-6  # Starting learning rate
  target_lr = 0.1  # Target learning rate after warm-up

  # Warm-Up Scheduler
  warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer,
      lr_lambda=lambda epoch: initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs) / target_lr if epoch < warmup_epochs else 1.0
  )

  # Step Decay Scheduler
  step_decay_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

  # Combine Schedulers
  scheduler = torch.optim.lr_scheduler.SequentialLR(
      optimizer,
      schedulers=[warmup_scheduler, step_decay_scheduler],
      milestones=[warmup_epochs]
  )

  # Learning rate scheduler: Reduce by factor of 10 every 50 epochs
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
  # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
  criterion = nn.CrossEntropyLoss()
  scaler = GradScaler()

  for epoch in range(epochs):
      start_epoch_time = time.time()  # Start time for the epoch
      print(f"Epoch {epoch + 1}/{epochs} - Training Started...")
      model.train()
      for inputs, labels in train_loader:
          inputs, labels = inputs.to(device), labels.to(device)
          # Warm-Up Phase
          if epoch < warmup_epochs:
              lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
              for param_group in optimizer.param_groups:
                  param_group['lr'] = lr

          # Regular Training Phase
          # (e.g., apply StepLR after warm-up)
          if epoch >= warmup_epochs and (epoch - warmup_epochs) % 50 == 0:
              for param_group in optimizer.param_groups:
                  param_group['lr'] *= 0.1  # Decay by factor of 10 every 50 epochs

          optimizer.zero_grad()
          with autocast('cuda'):  # Explicitly specify the device type
              outputs = model(inputs)
              loss = criterion(outputs, labels)
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()
      epoch_time = time.time() - start_epoch_time
      print(f"Epoch {epoch + 1} completed in {epoch_time:.4f} seconds")
      scheduler.step()


  return evaluate(model, val_loader, device)

def prepare_data(mode, dataset, erm_model=None, device='cpu'):
    targets = np.array(dataset.targets)

    if mode in ["hard", "easy"]:
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
        margins = []
        with torch.no_grad():
            erm_model.eval()
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = F.softmax(erm_model(inputs), dim=1)
                correct_class_probs = outputs[range(labels.size(0)), labels]
                max_wrong_class_probs = outputs.clone()
                max_wrong_class_probs[range(labels.size(0)), labels] = 0
                max_wrong_class_probs = max_wrong_class_probs.max(dim=1)[0]
                margin = correct_class_probs - max_wrong_class_probs
                margins.extend(margin.cpu().numpy())
        margins = np.array(margins)
        indices = np.argsort(margins)

        print(f"Margins calculated. Mode: {mode}")
        print("Top 5 margins (easy examples):", margins[indices[:5]])
        print("Bottom 5 margins (hard examples):", margins[indices[-5:]])

        if mode == "easy":
            val_idx = indices[:len(indices) // 10]
        elif mode == "hard":
            val_idx = indices[-len(indices) // 10:]
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx, val_idx = next(splitter.split(dataset.data, dataset.targets))

    train_idx = np.setdiff1d(np.arange(len(dataset)), val_idx)
    train_data = Subset(dataset, train_idx)
    val_data = Subset(dataset, val_idx)

    print(f"Mode: {mode}, Train size: {len(train_data)}, Val size: {len(val_data)}")
    return train_data, val_data

# Main script
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=12, pin_memory=True)

    # Train ERM
    print("Training ERM...")
    erm_model = WideResNet(depth=28, widen_factor=10, num_classes=100, dropout_rate=0.3).to(device)
    train_data, val_data = prepare_data("random", dataset)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=12, pin_memory=True)
    erm_accuracy = train(erm_model, train_loader, val_loader, device, epochs=100)
    print(f"ERM Test Accuracy: {erm_accuracy:.2f}%")

    # Train LRW variants
    results = {"erm": erm_accuracy}
    for mode in ["hard", "easy", "random"]:
        print(f"Training LRW-{mode.capitalize()}...")
        train_data, val_data = prepare_data(mode, dataset, erm_model=erm_model, device=device)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=12, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=12, pin_memory=True)
        model = WideResNet(depth=28, widen_factor=10, num_classes=100, dropout_rate=0.3).to(device)
        accuracy = train(model, train_loader, val_loader, device, epochs=100)
        results[mode] = accuracy
        print(f"LRW-{mode.capitalize()} Test Accuracy: {results[mode]:.2f}%")

# Plot gains
gains = {k: (v / results["erm"] * -1 if v / results["erm"] < 1 else v / results["erm"]) for k, v in results.items() if k != "erm"}
plt.bar(gains.keys(), gains.values())
plt.xlabel("LRW Method")
plt.ylabel("Accuracy Gain Over ERM (%)")
plt.title("Accuracy Gains on CIFAR-100")
plt.show()

# Define Splitter Network
class SplitterNetwork(nn.Module):
    def __init__(self, backbone, backbone_output_dim):
        super(SplitterNetwork, self).__init__()
        self.backbone = backbone  # Pretrained backbone (e.g., ResNet, EfficientNet)
        self.mlp = nn.Sequential(
            nn.Linear(backbone_output_dim, 128),  # Match backbone output dimension
            nn.ReLU(),
            nn.Linear(128, 1),  # Output single scalar for splitting decision
            nn.Sigmoid()  # Predict probability for splitting
        )

    def forward(self, x):
        with torch.no_grad():  # Keep backbone parameters fixed
            features = self.backbone(x)
        decision = self.mlp(features)  # Predict splitting decision
        return decision

  # Define Meta-Network
class MetaNetwork(nn.Module):
    def __init__(self, backbone, feature_dim):
        super(MetaNetwork, self).__init__()
        self.backbone = backbone  # Pretrained backbone (e.g., ResNet, EfficientNet)
        self.fc = nn.Linear(feature_dim, 1)  # Fully connected layer for predicting weights

    def forward(self, x):
        with torch.no_grad():  # Keep backbone parameters fixed
            features = self.backbone(x)
        weights = torch.sigmoid(self.fc(features))  # Predict weights in range [0, 1]
        return weights

def generate_split(train_loader, splitter_net, device, threshold=0.5):
    # Example logic to filter and create splits
    S_tr, S_val = [], []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        split_scores = splitter_net(x)  # Generate scores
        mask = split_scores > threshold
        if mask.sum() > 0:  # Ensure non-empty split
            S_val.append((x[mask], y[mask]))
        else:
            S_tr.append((x, y))  # Fallback to training set

    # Convert to DataLoader
    return DataLoader(S_tr), DataLoader(S_val)


def compute_reweighted_loss(data_loader, model, meta_net, loss_fn, device):
    model = model.to(device)
    meta_net = meta_net.to(device)
    total_loss = 0
    num_samples = 0

    for x, y in data_loader:
        # Skip empty batches
        if y.numel() == 0:
            print("Skipping empty batch...")
            continue

        x, y = x.to(device), y.to(device)
        outputs = model(x)  # Shape: (batch_size, num_classes)
        weights = meta_net(x).squeeze(-1)  # Shape: (batch_size,)

        # Ensure target shape
        if y.dim() == 2 and y.shape[1] == 1:
            y = y.squeeze(1)

        # Check batch size alignment
        if outputs.size(0) != y.size(0):
            raise ValueError(
                f"Batch size mismatch: outputs ({outputs.size(0)}), y ({y.size(0)})"
            )

        # Compute loss
        losses = loss_fn(outputs, y)
        reweighted_loss = (weights * losses).mean()
        total_loss += reweighted_loss.item()
        num_samples += x.size(0)

    if num_samples == 0:  # Handle case where all batches are skipped
        return 0
    return total_loss / len(data_loader)



def old_compute_reweighted_loss(data_loader, model, meta_net, loss_fn, device):
    """
    Compute the reweighted loss to train the meta-network.

    Args:
        data_loader: DataLoader for the validation set.
        model: Classifier model.
        meta_net: Meta-network predicting instance weights.
        loss_fn: Loss function (e.g., CrossEntropyLoss).
        device: Device ('cpu' or 'cuda') for computation.

    Returns:
        reweighted_loss: The reweighted loss for the meta-network.
    """
    model = model.to(device)  # Move model to the correct device
    meta_net = meta_net.to(device)  # Move meta-network to the correct device
    total_loss = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)  # Move input and labels to the correct device
        x = x.unsqueeze(0)
        outputs = model(x)  # Forward pass through the model
        weights = meta_net(x)  # Predict instance weights
        losses = loss_fn(outputs, y)  # Compute per-instance losses

        # Ensure weights and losses have compatible shapes
        weights = weights.squeeze()  # Remove extra dimensions from weights if necessary
        reweighted_loss = (weights * losses).mean()  # Compute weighted mean loss
        total_loss += reweighted_loss

    return total_loss

def compute_splitter_loss(S_tr, S_val, splitter_net, classifier, loss_fn, device):
    """
    Compute the splitter loss to train the splitter network.

    Args:
        S_tr: Training set (generated by the splitter network).
        S_val: Validation set (generated by the splitter network).
        splitter_net: The splitter network (used to predict splitting probabilities).
        classifier: The classifier being trained.
        loss_fn: The loss function (e.g., CrossEntropyLoss).

    Returns:
        splitter_loss: The loss for the splitter network.
    """
    # Compute training loss
    train_loss = 0
    for x_tr, y_tr in S_tr:
        outputs_tr = classifier(x_tr)
        train_loss += loss_fn(outputs_tr, y_tr).mean()
    train_loss /= len(S_tr)

    # Compute validation loss
    val_loss = 0
    for x_val, y_val in S_val:
        outputs_val = classifier(x_val)
        val_loss += loss_fn(outputs_val, y_val).mean()
    val_loss /= len(S_val)

    # Splitter loss: difference between validation and training loss
    splitter_loss = val_loss - train_loss
    return splitter_loss

import random

def probabilistic_split(data_loader, splitter_net, threshold=0.5):
    """
    Generate probabilistic splits for training and validation sets using the splitter network.

    Args:
        data_loader: DataLoader for the dataset.
        splitter_net: Splitter network predicting splitting probabilities.
        threshold: Probability threshold for splitting.

    Returns:
        S_tr: Training set.
        S_val: Validation set.
    """
    S_tr, S_val = [], []
    for x, y in data_loader:
        with torch.no_grad():
            split_probs = splitter_net(x)  # Predict splitting probabilities
        for i, prob in enumerate(split_probs):
            if random.random() < prob:  # Assign to training set with probability `prob`
                S_tr.append((x[i], y[i]))
            else:  # Otherwise assign to validation set
                S_val.append((x[i], y[i]))
    return S_tr, S_val

# Initialize networks
meta_net = MetaNetwork(erm_model, feature_dim=100)  # Example feature_dim
backbone_output_dim = 100  # Replace with the actual dimension from the previous step
splitter_net = SplitterNetwork(erm_model, backbone_output_dim)

# Optimizers
meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=0.001)
splitter_optimizer = torch.optim.Adam(splitter_net.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_fn = nn.CrossEntropyLoss(reduction='none')  # Compute per-instance losses

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Move models to device
splitter_net = splitter_net.to(device)
model = model.to(device)
meta_net = meta_net.to(device)

# Ensure data is on the same device
for epoch in range(100):
    S_tr, S_val = generate_split(train_loader, splitter_net, device, threshold=0.5)
    print(f"Training split: {len(S_tr)}, Validation split: {len(S_val)}")

    meta_optimizer.zero_grad()
    reweighted_loss = compute_reweighted_loss(S_val, model, meta_net, loss_fn, device)
    reweighted_loss.backward()
    meta_optimizer.step()

    # Step 3: Train the classifier using reweighted training loss
    optimizer.zero_grad()
    train_loss = compute_reweighted_loss(S_tr, model, meta_net, loss_fn)
    train_loss.backward()
    optimizer.step()

    # Step 4: Optionally update the splitter network
    splitter_optimizer.zero_grad()
    splitter_loss = compute_splitter_loss(S_tr, S_val, splitter_net, loss_fn)
    splitter_loss.backward()
    splitter_optimizer.step()

    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {reweighted_loss:.4f}")

