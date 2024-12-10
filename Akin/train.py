import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from load import load_data

class MOT20Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        return load_data(self.data_path)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        tracklets, gt_offsets = self.data
        item = [tracklets[idx], gt_offsets[idx]]

        if self.transform:
            item = self.transform(item)

        return item


class InteractionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, threshold=0.6):
        super(InteractionModule, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.asymmetric_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0)),
            nn.PReLU()
        )
        self.sigmoid = nn.Sigmoid()
        self.threshold = threshold
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # Predicting offsets (x, y, w, h)
        )

    def forward(self, tracklet_features):
        # Step 1: Embedding and attention computation
        embedded = self.embedding(tracklet_features)
        query = self.query(embedded)  # (batch_size, num_tracklets, hidden_dim)
        key = self.key(embedded)     # (batch_size, num_tracklets, hidden_dim)
        attention = torch.bmm(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)
        attention = F.softmax(attention, dim=-1)  # (batch_size, num_tracklets, num_tracklets)

        # Step 2: Prepare attention for convolution
        # Add a channel dimension and expand to (batch_size, hidden_dim, num_tracklets, num_tracklets)
        attention = attention.unsqueeze(1).repeat(1, embedded.size(-1), 1, 1)

        # Step 3: Asymmetric convolution for interaction modeling
        attention = self.asymmetric_conv(attention)

        # Step 4: Mask significant interactions
        interaction_mask = (self.sigmoid(attention) > self.threshold).float()
        interaction_matrix = interaction_mask * attention

        # Step 5: Motion prediction
        interaction_matrix = interaction_matrix.mean(dim=1)  # Reduce channel dimension back
        fused_features = torch.bmm(interaction_matrix, embedded)
        predicted_offsets = self.final_mlp(fused_features)

        return predicted_offsets

# Training Loop
def train_interaction_module(data_loader, model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for tracklet_features, ground_truth_offsets in data_loader:
            # Forward pass
            predicted_offsets = model(tracklet_features)

            # Compute loss
            loss = criterion(predicted_offsets, ground_truth_offsets)
            epoch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(data_loader):.4f}")

    print("Training completed.")


# Main
if __name__ == "__main__":
    # Device setup
    device = torch.device("mps")

    # Dataset and DataLoader
    # transform = transforms.Compose([transforms.ToTensor()])
    dataset = MOT20Dataset(data_path="gt.txt")
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Model, Loss, Optimizer
    interaction_module = InteractionModule(input_dim=8, hidden_dim=128)
    criterion = nn.MSELoss()  # Example loss function for offset prediction
    optimizer = optim.Adam(interaction_module.parameters(), lr=1e-3)

    # Train
    train_interaction_module(data_loader, interaction_module, criterion, optimizer, num_epochs=100)
