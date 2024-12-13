import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from load import load_data

class MOTDataset(Dataset):
    def __init__(self):
        self.data = self.load_data()

    def load_data(self):
        inputs, outputs = load_data('gt1.txt')
        inputs2, outputs2 = load_data('gt2.txt')
        inputs3, outputs3 = load_data('gt3.txt')
        inputs5, outputs5 = load_data('gt5.txt')

        inputs = inputs + inputs2 + inputs3 + inputs5
        outputs = outputs + outputs2 + outputs3 + outputs5

        return inputs, outputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tracklets, outputs = self.data

        return tracklets[idx], outputs[idx]

class InteractionModule(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=4):
        super(InteractionModule, self).__init__()
        # Linear transformation for embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)

        self.prelu = nn.PReLU()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=(1, 0))

        # Graph Convolution Network
        self.gcn = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.PReLU()
        )

        self.prediction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, inputs):
        o = inputs[:,:,:4]

        x = self.embedding(inputs)

        q = self.query(x)
        k = self.key(x)

        attention_scores = F.softmax(torch.bmm(q, k.transpose(1, 2)) / np.sqrt(k.size(-1)), dim=-1)

        conv_features = attention_scores
        for _ in range(10):
            conv_features = self.conv1(conv_features) + self.conv2(conv_features)
            conv_features = self.prelu(conv_features)

        mask = torch.sigmoid(conv_features)
        mask = (mask > 0.5).float()

        adjacency_matrix = mask * attention_scores

        gcn_features = self.gcn(torch.mm(adjacency_matrix.squeeze(0), o.squeeze(0)).unsqueeze(0))

        predictions = self.prediction(gcn_features)

        return predictions

def validate_model(model, val_dataloader, criterion, device='cuda'):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            predictions = model(inputs)

            loss = criterion(predictions, targets)

            val_loss += loss.item()

    return val_loss

# Define the training loop
def train_model(model, dataloader, val_dataloader, optimizer, criterion, num_epochs=10, device='cuda'):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            predicted_offsets = model(inputs)

            # Compute loss
            loss = criterion(predicted_offsets, targets)

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()


        val_loss = validate_model(model, val_dataloader, criterion, device=device)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(dataloader):.4f}, Val Loss: {val_loss/len(val_dataloader):.4f}")
        # print(f"{total_loss/len(dataloader):.4f}")

# Example usage
if __name__ == "__main__":
    train_dataset = MOTDataset()
    val_dataset = MOTDataset()
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = InteractionModule()
    criterion = nn.MSELoss()  # Example loss function for offset prediction
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=40, device='mps')
