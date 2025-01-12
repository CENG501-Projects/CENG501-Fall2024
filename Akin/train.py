import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from load import load_data
from plot import plot_loss

class InteractionLoss(nn.Module):
    def __init__(self):
        super(InteractionLoss, self).__init__()

    def forward(self, pred, target):
        pred_sizes= pred[:, 2:]
        target_sizes = target[:, 2:]

        pred2 = pred[:, :2] + pred[:, 2:]
        target2 = target[:, :2] + target[:, 2:]

        x1 = torch.max(pred[:, 0], target[:, 0])
        y1 = torch.max(pred[:, 1], target[:, 1])
        x2 = torch.min(pred2[:, 0], target2[:, 0])
        y2 = torch.min(pred2[:, 1], target2[:, 1])

        inter_width = torch.clamp(x2 - x1, min=0)
        inter_height = torch.clamp(y2 - y1, min=0)
        intersection = inter_width * inter_height

        pred_area = pred_sizes[:, 0] * pred_sizes[:, 1]
        target_area = target_sizes[:, 0] * target_sizes[:, 1]

        union = torch.clamp(pred_area + target_area - intersection, min=0)

        epsilon = 1e-7 # to avoid division by zero
        iou = (intersection + epsilon) / (union + epsilon)

        loss = 1 - torch.square(iou)
        return loss.mean()

class AsymmetricConvSumLayer(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = 1
        out_channels = 1
        K = 3

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, K),
            padding=(0, K//2),
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(K, 1),
            padding=(K//2, 0),
            bias=False
        )
        self.prelu = nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        out = self.conv1(x) + self.conv2(x)
        return self.prelu(out)

class InteractionModule(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=4):
        super(InteractionModule, self).__init__()
        F = hidden_dim
        D = hidden_dim

        # Linear transformation for embedding
        self.embedding = nn.Linear(input_dim, F)

        self.query = nn.Linear(F, D)
        self.key = nn.Linear(F, D)

        conv_layers = []
        for _ in range(10):
            conv_layers.append(AsymmetricConvSumLayer())

        self.conv_network = nn.Sequential(*conv_layers)

        # Graph Convolution Network
        self.gcn = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.PReLU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, inputs):
        o = inputs[:,4:]

        x = self.embedding(inputs)

        q = self.query(x)
        k = self.key(x)

        p1 = k.transpose(0, 1)
        p2 = torch.mm(q, p1)
        p3 = p2 / np.sqrt(k.size(-1))
        attention_scores = F.softmax(p3, dim=-1)

        conv_input = attention_scores.unsqueeze(0)
        conv_output = self.conv_network(conv_input)
        conv_features = conv_output.squeeze(0)

        mask = torch.sigmoid(conv_features)
        mask = (mask > 0.5).float()

        adjacency_matrix = mask * attention_scores

        calc = torch.mm(adjacency_matrix, o)
        gcn_features = self.gcn(calc)

        predictions = self.mlp(gcn_features)

        return predictions

class RefindModule(nn.Module):
    def __init__(self):
        super(RefindModule, self).__init__()

        self.linear1 = nn.Linear(5, 64)

        in_channels = 1
        out_channels = 1
        K = 3

        conv_layers = []
        for _ in range(10):
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(K, 1),
                padding=(K//2, 0),
                bias=False
            )
            conv_layers.append(
                nn.Sequential(conv, nn.PReLU())
            )
        self.conv1 = nn.Sequential(*conv_layers)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, K),
                padding=(0, K//2),
                bias=False
            ),
            nn.PReLU(),
        )
        self.pool = nn.AdaptiveMaxPool2d((64, 1))

        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, Tlost, Drest):
        TlostP = Tlost[:,-1,:] # this is Sx5
        # print('S is', TlostP.shape[0])
        # print('U is', Drest.shape[0])

        DrestP = Drest.unsqueeze(0) # this is 1xUx5
        TlostP = TlostP.unsqueeze(1) # this is Sx1x5
        DrestP = DrestP - TlostP # this is SxUx5

        Fdete = self.linear1(DrestP)

        TL = Tlost.unsqueeze(1)
        TL = self.conv1(TL)

        Ftraj = self.conv2(TL)
        Ftraj = Ftraj.squeeze(1)

        Ftraj = self.pool(Ftraj)
        Ftraj = Ftraj.squeeze(2)

        Ftraj = Ftraj.unsqueeze(1)
        Ftraj = Ftraj.repeat(1, Fdete.shape[1], 1)

        R = torch.cat((Ftraj, Fdete), dim=2)
        # print(R.shape)

        R = self.mlp(R)
        R = R.squeeze(-1)

        R = torch.sigmoid(R)

        return R


def train_model(dataloader, interaction_model, refind_model, interaction_optimizer, refind_optimizer, interaction_criterion, refind_criterion, num_epochs=10, device='cuda'):
    interaction_train_losses = []
    refind_train_losses = []

    interaction_model.to(device)
    refind_model.to(device)

    for epoch in range(num_epochs):
        interaction_epoch_loss = 0
        refind_epoch_loss = 0

        interaction_model.train()
        refind_model.train()

        for batch in dataloader:
            last_track_ids = set()
            last_locations = dict()
            lost_track_ids = set()

            input_group = batch[0]
            output_group = batch[1]

            for frame_index in range(len(input_group)):
                inputs = input_group[frame_index]
                outputs = output_group[frame_index]

                index_tensor = torch.tensor([frame_index]).to(device)

                inputs, outputs = inputs.to(device), outputs.to(device)

                interaction_inputs = inputs[:, 1:]
                interaction_outputs = outputs

                track_ids = inputs[:, 0]
                track_ids = [int(x.item()) for x in track_ids]
                track_ids = set(track_ids)

                new_track_ids = track_ids - last_track_ids
                lost_track_ids = lost_track_ids.union(last_track_ids - track_ids)
                last_track_ids = track_ids

                interaction_optimizer.zero_grad()
                predicted_offsets = interaction_model(interaction_inputs)
                new_locations = interaction_inputs[:,:4] + predicted_offsets

                interaction_loss = interaction_criterion(new_locations, interaction_outputs)

                interaction_epoch_loss += interaction_loss.item()
                interaction_loss.backward()
                interaction_optimizer.step()

                for i, track_id in enumerate(track_ids):
                    if track_id not in last_locations:
                        last_locations[track_id] = [torch.tensor([0 for _ in range(5)], dtype=torch.float).to(device) for _ in range(30)]

                    last_location = torch.cat((index_tensor, interaction_outputs[i]), dim=-1)
                    last_locations[track_id].append(last_location)

                    if len(last_locations[track_id]) > 30:
                        last_locations[track_id].pop(0)

                if len(lost_track_ids) == 0 or len(new_track_ids) == 0:
                    continue

                lost_track_ids_list = list(lost_track_ids)
                new_track_ids_list = list(new_track_ids)
                Tlost = torch.stack([torch.stack(last_locations[track_id]) for track_id in lost_track_ids_list])
                Drest = torch.stack([last_locations[track_id][-1] for track_id in new_track_ids_list])

                refind_outputs = torch.tensor([0 for _ in range(Drest.shape[0])], dtype=torch.float).to(device)
                refind_outputs = refind_outputs.repeat(Tlost.shape[0], 1)

                for i, track_id in enumerate(new_track_ids_list):
                    if track_id not in lost_track_ids:
                        continue
                    lost_track_id = lost_track_ids_list.index(track_id)
                    refind_outputs[lost_track_id][i] = 1.0

                refind_optimizer.zero_grad()
                refind_predictions = refind_model(Tlost, Drest)
                refind_loss = refind_criterion(refind_predictions, refind_outputs)

                refind_epoch_loss += refind_loss.item()
                refind_loss.backward()
                refind_optimizer.step()

                lost_track_ids = lost_track_ids - track_ids

        interaction_train_loss = interaction_epoch_loss
        interaction_train_losses.append(interaction_train_loss)

        refind_train_loss = refind_epoch_loss
        refind_train_losses.append(refind_train_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss Interaction: {interaction_train_loss:.4f}, Train Loss Refind: {refind_train_loss:.4f}")

    return interaction_train_losses

# Example usage
if __name__ == "__main__":
    dataset = [
        load_data('gt1.txt'),
        load_data('gt2.txt'),
        load_data('gt3.txt'),
        load_data('gt5.txt'),
    ]

    interaction_criterion = InteractionLoss()
    refind_criterion = nn.BCELoss()

    interaction_model = InteractionModule()
    refind_model = RefindModule()

    interaction_optimizer = optim.Adam(interaction_model.parameters(), lr=1e-3)
    refind_optimizer = optim.Adam(refind_model.parameters(), lr=1e-3)

    train_losses = train_model(dataset, interaction_model, refind_model, interaction_optimizer, refind_optimizer, interaction_criterion, refind_criterion, num_epochs=10, device='cpu')
    plot_loss(train_losses)
