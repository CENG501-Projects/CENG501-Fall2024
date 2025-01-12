import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import os
import math
import csv

# ----------------------------
#   Generator (same as FLGAN)
# ----------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(noise_dim, 7 * 7 * 256),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 5, stride=2, padding=2, output_padding=1)
            # NOTE: no final Sigmoid, will use BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.main(x)


# -------------------------------
#   Discriminator (same as FLGAN)
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(7 * 7 * 128, 1)
            # No sigmoid => BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.main(x)


# -------------------------------
#  Create K=2 data splits (MNIST)
# -------------------------------
def create_data_splits_k2(dataset, case):
    """
    Splits MNIST into two subsets (client1: digits 0-4, client2: digits 5-9).
    The 'case' argument can be:
      - "balanced": 10k samples each
      - "imbalanced": 10k for client1, 1k for client2
      - "extremely_imbalanced": 10k for client1, 100 for client2
    """
    client1_indices = [i for i, (x, y) in enumerate(dataset) if y in {0, 1, 2, 3, 4}]
    client2_indices = [i for i, (x, y) in enumerate(dataset) if y in {5, 6, 7, 8, 9}]

    if case == "balanced":
        client1_indices = client1_indices[:10000]
        client2_indices = client2_indices[:10000]
    elif case == "imbalanced":
        client1_indices = client1_indices[:10000]
        client2_indices = client2_indices[:1000]
    elif case == "extremely_imbalanced":
        client1_indices = client1_indices[:10000]
        client2_indices = client2_indices[:100]
    else:
        raise ValueError("Invalid case.")

    client1_data = Subset(dataset, client1_indices)
    client2_data = Subset(dataset, client2_indices)
    return client1_data, client2_data


# -------------------------------
#  Create K=5 data splits (MNIST)
# -------------------------------
def create_data_splits_k5(dataset, case):
    """
    Splits MNIST into five subsets:
      client1: digits {0,1}
      client2: digits {2,3}
      client3: digits {4,5}
      client4: digits {6,7}
      client5: digits {8,9}

    The 'case' argument can be:
      - "balanced": 10k samples each client (if enough samples exist)
      - "imbalanced": 10k for client1, 1k for others (if enough samples exist)
      - "extremely_imbalanced": 10k for client1, 100 for others
    """
    # Define digit sets for each client
    digit_groups = [
        {0, 1},  # client1
        {2, 3},  # client2
        {4, 5},  # client3
        {6, 7},  # client4
        {8, 9},  # client5
    ]

    # Collect indices per client
    all_client_indices = []
    for dg in digit_groups:
        idx = [i for i, (x, y) in enumerate(dataset) if y in dg]
        all_client_indices.append(idx)

    # For each client, slice the appropriate number of samples
    def get_slice(case, client_idx):
        if case == "balanced":
            return 10000  # up to 10k per client
        elif case == "imbalanced":
            # client0 => 10k, all others => 1k
            return 10000 if client_idx == 0 else 1000
        elif case == "extremely_imbalanced":
            return 10000 if client_idx == 0 else 100
        else:
            raise ValueError("Invalid case.")

    subsets = []
    for i, idx_list in enumerate(all_client_indices):
        limit = get_slice(case, i)
        idx_list = idx_list[:limit]
        subsets.append(Subset(dataset, idx_list))

    return subsets  # list of length 5


# -------------------------------
#  Create K=10 data splits (MNIST)
# -------------------------------
def create_data_splits_k10(dataset, case):
    """
    Splits MNIST into ten subsets, each containing exactly one digit:
      client1: digit {0}
      client2: digit {1}
      ...
      client10: digit {9}

    The 'case' argument can be:
      - "balanced": up to 10k samples per digit
      - "imbalanced": client1 => 10k, all others => 1k
      - "extremely_imbalanced": client1 => 10k, all others => 100
    """
    digit_groups = [
        {0}, {1}, {2}, {3}, {4}, 
        {5}, {6}, {7}, {8}, {9}
    ]

    all_client_indices = []
    for dg in digit_groups:
        idx = [i for i, (x, y) in enumerate(dataset) if y in dg]
        all_client_indices.append(idx)

    def get_slice(case, client_idx):
        if case == "balanced":
            return 10000
        elif case == "imbalanced":
            return 10000 if client_idx == 0 else 1000
        elif case == "extremely_imbalanced":
            return 10000 if client_idx == 0 else 100
        else:
            raise ValueError("Invalid case.")

    subsets = []
    for i, idx_list in enumerate(all_client_indices):
        limit = get_slice(case, i)
        idx_list = idx_list[:limit]
        subsets.append(Subset(dataset, idx_list))

    return subsets  # list of length 10


# --------------------------------
#  (Optional) RBF-based MMD Score
# --------------------------------
def compute_mmd_rbf(x, y, sigma=1.0):
    """
    A simple implementation of MMD using an RBF kernel:
    MMD^2 = E[k(x, x')] + E[k(y, y')] - 2E[k(x, y)]
    Where k(a, b) = exp(-||a - b||^2 / (2*sigma^2)).

    Returns the MMD (not squared).
    """
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    if y.dim() > 2:
        y = y.view(y.size(0), -1)

    xx = torch.mm(x, x.t())  # shape (B,B)
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    x2 = (x * x).sum(dim=1).unsqueeze(0)
    y2 = (y * y).sum(dim=1).unsqueeze(0)

    # dist(a,b) = ||a||^2 + ||b||^2 - 2 a.b
    xx_dist = x2 + x2.t() - 2.0 * xx
    yy_dist = y2 + y2.t() - 2.0 * yy
    xy_dist = x2 + y2.t() - 2.0 * xy

    k_xx = torch.exp(-xx_dist / (2 * sigma**2))
    k_yy = torch.exp(-yy_dist / (2 * sigma**2))
    k_xy = torch.exp(-xy_dist / (2 * sigma**2))

    m = x.size(0)
    n = y.size(0)
    E_xx = k_xx.sum() / (m*m)
    E_yy = k_yy.sum() / (n*n)
    E_xy = k_xy.sum() / (m*n)

    mmd_sq = E_xx + E_yy - 2.0 * E_xy
    return torch.sqrt(torch.clamp(mmd_sq, min=1e-9))


# --------------------------------
#   Save Generated Images (same)
# --------------------------------
def generate_and_save_images(generator, epoch, noise, folder="ifl-gan-images"):
    os.makedirs(folder, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        fake_images = generator(noise).cpu()
        # Optionally clamp or tanh if you want them strictly in [-1,1]
        fake_images = torch.tanh(fake_images)
        fake_images = fake_images.view(fake_images.size(0), 28, 28)

        fig = plt.figure(figsize=(4, 4))
        for i in range(fake_images.size(0)):
            plt.subplot(4, 4, i+1)
            plt.imshow(fake_images[i], cmap='gray')
            plt.axis('off')

        plt.savefig(f"{folder}/epoch_{epoch+1:03d}.png")
        plt.close()


# -----------------------------
#   IFL-GAN Implementation
# -----------------------------
class IFLGAN:
    """
    IFL-GAN (Algorithm 1) with local training + MMD-based aggregator + threshold update:
      1) Each client trains local (G_i, D_i) on real/fake.
      2) Compute MMD_i for each client i.
      3) Convert MMD_i -> alpha_i via softmax.
      4) G_glb = sum_i alpha_i * G_i
      5) If MMD_i > threshold, G_i <- G_glb (partial "consensus" update)
    """
    def __init__(
        self, 
        noise_dim=128, 
        case="balanced", 
        num_clients=2, 
        batch_size=128, 
        lr=0.0002, 
        device=None
    ):
        self.noise_dim = noise_dim
        self.num_clients = num_clients
        self.batch_size = batch_size

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Build local G & D
        self.generators = [Generator(noise_dim).to(self.device) for _ in range(num_clients)]
        self.discriminators = [Discriminator().to(self.device) for _ in range(num_clients)]

        # Prepare MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # scale to [-1,1]
        ])
        mnist_dataset = torchvision.datasets.MNIST(
            root="./data", 
            train=True, 
            download=True, 
            transform=transform
        )

        # According to the number of clients, split dataset
        if num_clients == 2:
            # K=2 case
            c1_data, c2_data = create_data_splits_k2(mnist_dataset, case)
            self.local_datasets = [c1_data, c2_data]
        elif num_clients == 5:
            # K=5 case
            subsets = create_data_splits_k5(mnist_dataset, case)
            self.local_datasets = subsets
        elif num_clients == 10:
            # K=10 case
            subsets = create_data_splits_k10(mnist_dataset, case)
            self.local_datasets = subsets
        else:
            raise NotImplementedError(f"This example handles exactly K=2, K=5, or K=10 (got {num_clients}).")

        # Create local dataloaders
        self.local_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
            for ds in self.local_datasets
        ]

        # Create optimizers
        betas = (0.5, 0.999)
        self.optimizers_gen = [
            optim.Adam(g.parameters(), lr=lr, betas=betas) 
            for g in self.generators
        ]
        self.optimizers_disc = [
            optim.Adam(d.parameters(), lr=lr, betas=betas) 
            for d in self.discriminators
        ]

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

    def train_discriminator(self, real_images, generator, discriminator, disc_optimizer):
        """
        One step of training the local discriminator:
          - Real images => label 0.9
          - Fake images => label 0.0
        """
        disc_optimizer.zero_grad()

        real_images = real_images.to(self.device)
        real_labels = torch.full((real_images.size(0), 1), 0.9, device=self.device)
        real_logits = discriminator(real_images)
        loss_real = self.criterion(real_logits, real_labels)

        noise = torch.randn(real_images.size(0), self.noise_dim, device=self.device)
        fake_images = generator(noise)
        fake_labels = torch.zeros((real_images.size(0), 1), device=self.device)
        fake_logits = discriminator(fake_images.detach())
        loss_fake = self.criterion(fake_logits, fake_labels)

        loss_d = loss_real + loss_fake
        loss_d.backward()
        disc_optimizer.step()
        return loss_d.item()

    def train_generator(self, generator, discriminator, gen_optimizer):
        """
        One step of training the local generator:
          - Fake images => label 1.0 (fool the discriminator)
        """
        gen_optimizer.zero_grad()

        noise = torch.randn(self.batch_size, self.noise_dim, device=self.device)
        fake_images = generator(noise)

        labels = torch.ones((fake_images.size(0), 1), device=self.device)  # wants to fool D
        logits = discriminator(fake_images)
        loss_g = self.criterion(logits, labels)

        loss_g.backward()
        gen_optimizer.step()
        return loss_g.item()

    def compute_local_mmd(self, generator, data_loader, num_samples=64, sigma=1.0):
        """
        Sample 'num_samples' real images + generate 'num_samples' fakes => compute MMD.
        """
        real_iter = iter(data_loader)
        try:
            real_images, _ = next(real_iter)
        except StopIteration:
            # If the loader is exhausted, re-initialize it
            real_iter = iter(data_loader)
            real_images, _ = next(real_iter)

        # Trim real images to num_samples
        real_images = real_images[:num_samples].to(self.device)

        noise = torch.randn(num_samples, self.noise_dim, device=self.device)
        with torch.no_grad():
            fake_images = generator(noise)

        mmd_val = compute_mmd_rbf(real_images, fake_images, sigma=sigma)
        return mmd_val.item()

    def aggregate_global_generator(self, mmd_values):
        """
        1) Convert MMD_i -> alpha_i via softmax
        2) Weighted sum of local generator params => G_glb
        3) Return G_glb (as a state_dict)
        """
        mmd_tensor = torch.tensor(mmd_values, dtype=torch.float32, device=self.device)
        exps = torch.exp(mmd_tensor)  # e^(mmd_i)
        alpha = exps / torch.sum(exps)  # softmax

        global_params = {}
        # We'll combine parameters from all local generators
        for key in self.generators[0].state_dict():
            stacked = []
            for i in range(self.num_clients):
                stacked.append(
                    self.generators[i].state_dict()[key].float() * alpha[i]
                )
            global_params[key] = torch.stack(stacked, dim=0).sum(dim=0)

        return global_params, alpha

    def ifl_training(self, epochs=30, sigma=1.0):
        """
        Main loop for IFL-GAN:
          1) Local D/G training
          2) Compute MMD -> alpha_i -> aggregator
          3) threshold => update local G if MMD_i is above threshold
        """
        fixed_noise = torch.randn(16, self.noise_dim, device=self.device)

        # Track generator loss (averaged across clients each epoch)
        generator_losses = []

        for epoch in range(epochs):
            print(f"\nEpoch [{epoch+1}/{epochs}]")

            # ---- 1) Local training for each client
            epoch_gen_loss = 0.0
            for i in range(self.num_clients):
                gen_i = self.generators[i]
                disc_i = self.discriminators[i]
                opt_g = self.optimizers_gen[i]
                opt_d = self.optimizers_disc[i]

                loader_i = self.local_loaders[i]

                total_d, total_g = 0.0, 0.0
                for real_images, _ in loader_i:
                    d_loss = self.train_discriminator(real_images, gen_i, disc_i, opt_d)
                    g_loss = self.train_generator(gen_i, disc_i, opt_g)
                    total_d += d_loss
                    total_g += g_loss

                avg_d = total_d / len(loader_i)
                avg_g = total_g / len(loader_i)
                epoch_gen_loss += avg_g

                print(f"  Client {i+1} => D-loss: {avg_d:.4f}, G-loss: {avg_g:.4f}")

            # Average generator loss across all clients
            epoch_gen_loss /= self.num_clients
            generator_losses.append(epoch_gen_loss)

            # ---- 2) Compute MMD for each client
            mmd_scores = []
            for i in range(self.num_clients):
                mmd_val = self.compute_local_mmd(
                    self.generators[i], 
                    self.local_loaders[i],
                    num_samples=64, 
                    sigma=sigma
                )
                mmd_scores.append(mmd_val)

            # ---- 3) Aggregator: alpha_i via softmax of MMD, build G_glb
            global_params, alpha = self.aggregate_global_generator(mmd_scores)

            # ---- 4) Threshold check => each client updates local G if mmd_i > threshold
            threshold = sum(mmd_scores) / len(mmd_scores)  # Example: mean of MMD
            for i in range(self.num_clients):
                if mmd_scores[i] > threshold:
                    self.generators[i].load_state_dict(global_params)

            alpha_str = ", ".join([f"alpha_{i+1}={alpha[i]:.3f}" for i in range(self.num_clients)])
            print(f"  MMD scores: {mmd_scores}, threshold={threshold:.4f}")
            print(f"  Softmax => {alpha_str}")
            print(f"  [Global] Epoch {epoch+1} Avg Generator Loss: {epoch_gen_loss:.4f}")

            # ---- 5) Generate & save images from client #0 (arbitrary choice)
            if (epoch + 1) % 5 == 0 or (epoch == epochs - 1):
                generate_and_save_images(self.generators[0], epoch, fixed_noise)

        # =============================
        # Save generator losses to CSV
        # =============================
        with open("iflgan_generator_losses.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "gen_loss"])
            for ep_idx, loss_val in enumerate(generator_losses, start=1):
                writer.writerow([ep_idx, loss_val])

        print("\nIFL-GAN training done.")
        print("Saved generator losses to iflgan_generator_losses.csv")


# -----------------------------
#       Main Script
# -----------------------------
if __name__ == "__main__":
    # Example usage:
    ifl_gan = IFLGAN(
        noise_dim=128, 
        case="balanced",     # or "imbalanced", "extremely_imbalanced"
        num_clients=5,       # 2, 5, or 10
        batch_size=128, 
        lr=0.0002
    )
    ifl_gan.ifl_training(epochs=200, sigma=1.0)
