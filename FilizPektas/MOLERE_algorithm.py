from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import time
import copy
from utils import *

# compute_marginal_z_loss and compute_y_given_z_loss are taken from https://github.com/fl65inc/learning-to-split/blob/142c03912ee158da19cb74ad32230a3ce03ade7d/ls/training/utils.py#L5
# authors claim to use these methods for their experiments
def compute_marginal_z_loss(mask, tar_ratio, no_grad=False):
    '''
        Compute KL div between the splitter's z marginal and the prior z margional
        Goal: the predicted training size need to be tar_ratio * total_data_size
    '''
    cur_ratio = torch.mean(mask)
    cur_z = torch.stack([cur_ratio, 1 - cur_ratio])  # train split, test_split

    tar_ratio = torch.ones_like(cur_ratio) * tar_ratio
    tar_z = torch.stack([tar_ratio, 1 - tar_ratio])

    loss_ratio = F.kl_div(torch.log(cur_z), tar_z, reduction='batchmean')

    if not torch.isfinite(loss_ratio):
        loss_ratio = torch.ones_like(loss_ratio)

    if no_grad:
        loss_ratio = loss_ratio.item()

    return loss_ratio, cur_ratio.item()

def compute_y_given_z_loss(mask, y, no_grad=False):
    '''
      conditional marginal p(y | z = 1) need to match p(y | z = 0)
    '''
    # get num of classes
    num_classes = len(torch.unique(y))

    y_given_train, y_given_test, y_original = [], [], []

    for i in range(num_classes):
        y_i = (y == i).float()

        y_given_train.append(torch.sum(y_i * mask) / torch.sum(mask))
        y_given_test.append(torch.sum(y_i * (1 - mask)) / torch.sum(1 - mask))
        y_original.append(torch.sum(y_i) / len(y))

    y_given_train = torch.stack(y_given_train)
    y_given_test = torch.stack(y_given_test)
    y_original = torch.stack(y_original).detach()

    loss_y_marginal = F.kl_div(torch.log(y_given_train), y_original,
                               reduction='batchmean') + \
        F.kl_div(torch.log(y_given_test), y_original, reduction='batchmean')

    if not torch.isfinite(loss_y_marginal):
        loss_y_marginal = torch.ones_like(loss_y_marginal)

    if no_grad:
        loss_y_marginal = loss_y_marginal.item()

    return loss_y_marginal, y_given_train.tolist()[-1], y_given_test.tolist()[-1]

# Generate split
def generate_split(dataset, splitter, device, warm_up_count, delta=0.1, min_batch_size=2, warm_up_epoch_limit=26):
    splitter.eval()
    train_data, val_data = [], []
    loss_ratio = 0
    loss_balance = 0

    with torch.no_grad():
        if warm_up_count < warm_up_epoch_limit:
            val_size = int(len(dataset) * delta)
            train_size = len(dataset) - val_size

            # Perform a random split
            train_data, val_data = random_split(dataset, [train_size, val_size])
        else:
            # Assuming 'dataset' is a PyTorch Dataset
            data_loader = DataLoader(dataset, batch_size=8, shuffle=True)  # Example batch size

            for idx, (x, y) in enumerate(data_loader):
                x, y = x.to(device), y.to(device)

                # Use splitter network for dynamic splitting
                split_probs = splitter(x).squeeze()  # Predicted probabilities

                # Create a mask for the validation set
                val_mask = split_probs < 0.5  # Boolean mask for validation samples
                train_mask = ~val_mask        # Invert mask for training samples

                # Split x and y based on the mask
                val_data.extend(zip(x[val_mask].cpu(), y[val_mask].cpu()))
                train_data.extend(zip(x[train_mask].cpu(), y[train_mask].cpu()))

            # Compute additional losses for splitting strategy
            loss_ratio, ratio = compute_marginal_z_loss(split_probs, delta)
            loss_balance, ptrain_y, ptest_y = compute_y_given_z_loss(split_probs, y)

            # Adjust validation set size to match delta
            total_train = len(train_data)
            target_val_size = int(total_train * delta)  # Calculate desired val size

            if len(val_data) > target_val_size:
                val_data = val_data[:target_val_size]  # Directly slice the list

    print(f"Generated split: Train size = {len(train_data)}, Validation size = {len(val_data)}")

    # Check batch sizes
    if len(train_data) < min_batch_size or len(val_data) < min_batch_size:
        print("Could not find an eligible distribution, recalculating...")
        # Recalculate margins for hard/easy examples

        val_size = int(len(dataset) * delta)
        train_size = len(dataset) - val_size
        train_data, val_data = random_split(dataset, [train_size, val_size])

    return train_data, val_data, loss_ratio, loss_balance

# Train MOLERE framework
def train_molere_algorithm(
    dataset,
    classifier,
    meta_network,
    splitter,
    optimizer_classifier,
    optimizer_meta,
    optimizer_splitter,
    device,
    epochs=10,
    delta=0.1,
    q_updates=5,
    batch_size=8,
    warm_up_epoch_limit=26
):
    criterion = nn.CrossEntropyLoss()
    ge = 0.0
    warm_up_count = 0
    q_update_count = 0
    for epoch in range(epochs):
        start_time = time.time()  # Start timing the epoch
        print(f"Epoch {epoch + 1}/{epochs}")
        warm_up_count += 1
        q_update_count += 1

        # Generate train and validation splits
        train_data, val_data, loss_ratio, loss_balance = generate_split(dataset, splitter, device, warm_up_count, delta, warm_up_epoch_limit)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        classifier.train()
        meta_network.train()
        splitter.train()

        if warm_up_count >= warm_up_epoch_limit:
          # Update Meta and Splitter Networks
          for batch_idx, (x_val, y_val) in enumerate(val_loader):
              x_val, y_val = x_val.to(device), y_val.to(device)

              # Splitter update
              optimizer_splitter.zero_grad()
              val_probs = splitter(x_val).squeeze()
              y_val = y_val.argmax(dim=1) if y_val.dim() > 1 else y_val  # Convert one-hot to class indices if needed


              try:
                real_splitter_loss = criterion(val_probs, y_val.float())
              except:
                print(f"y_val: {y_val.shape}, val probs: {val_probs.shape}")
                print(x_val.shape)

              splitter_loss = (real_splitter_loss + loss_ratio + loss_balance) / 3
              splitter_loss.backward(retain_graph=True)
              optimizer_splitter.step()

              # Meta-Network update
              optimizer_meta.zero_grad()
              instance_weights = meta_network(x_val).squeeze()
              weighted_loss = (instance_weights * criterion(classifier(x_val), y_val)).mean()
              weighted_loss.backward(retain_graph=True)
              optimizer_meta.step()

        # Update Classifier
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            for _ in range(q_updates):
                optimizer_classifier.zero_grad()
                logits = classifier(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward(retain_graph=True)
                optimizer_classifier.step()

        # Generalization error
        train_loss = sum(
            criterion(classifier(x.to(device)), y.to(device)).item() for x, y in train_loader
        ) / len(train_loader)

        val_loss = sum(
            criterion(classifier(x.to(device)), y.to(device)).item() for x, y in val_loader
        ) / len(val_loader)

        ge = val_loss - train_loss
        print(f"Epoch {epoch + 1}: Generalization Error = {ge:.4f}")

        # Log epoch runtime
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} Runtime: {epoch_time:.2f} seconds")
        if abs(ge) < 1e-4:
          return classifier

    return classifier


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Define Splitter Network
class SplitterNetwork(nn.Module):
    def __init__(self, backbone, dataset_name):
        super(SplitterNetwork, self).__init__()

        # Move the backbone to CPU before deepcopy to avoid CUDA issues
        device = next(backbone.parameters()).device  # Get the current device of the backbone
        backbone_cpu = backbone.to('cpu')  # Temporarily move to CPU for deepcopy
        self.backbone = copy.deepcopy(backbone_cpu)  # Perform deepcopy on the CPU version
        self.backbone = self.backbone.to(device)  # Move the copied backbone back to the original device

        # Freeze the backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # MLP layer for splitting decision
        self.mlp = nn.Sequential(
            nn.Linear(self.backbone.fc.out_features, 1),  # Output single value for sigmoid
            nn.Sigmoid()  # Predict splitting decision
        ) if dataset_name != "cifar100" else  nn.Sequential(
            nn.Linear(self.backbone.linear.out_features, 1),  # Output single value for sigmoid
            nn.Sigmoid()  # Predict splitting decision
        )

    def forward(self, x):
        features = self.backbone(x)
        decision = self.mlp(features)
        return decision

# Define Meta-Network
class MetaNetwork(nn.Module):
    def __init__(self, backbone, dataset_name, output_dim=1):
        super(MetaNetwork, self).__init__()
        self.backbone = copy.deepcopy(backbone)
        self.new_fc = nn.Linear(backbone.fc.out_features, output_dim) if dataset_name != "cifar100" else nn.Linear(backbone.linear.out_features, output_dim)

    def forward(self, x):
        features = self.backbone(x)
        out = self.new_fc(features)
        return out
    