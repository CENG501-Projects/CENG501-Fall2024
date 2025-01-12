import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# ------------------------
#   Data Split Functions
# ------------------------
def create_data_splits_k2(dataset, case):
    """
    Splits MNIST into two subsets (clients):
      - Client 1: Digits {0,1,2,3,4}
      - Client 2: Digits {5,6,7,8,9}
    """
    client1_indices = [i for i, (_, y) in enumerate(dataset) if y in {0, 1, 2, 3, 4}]
    client2_indices = [i for i, (_, y) in enumerate(dataset) if y in {5, 6, 7, 8, 9}]

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
        raise ValueError("Invalid case (choose 'balanced', 'imbalanced', or 'extremely_imbalanced').")

    c1_data = Subset(dataset, client1_indices)
    c2_data = Subset(dataset, client2_indices)
    return [c1_data, c2_data]


def create_data_splits_k5(dataset, case):
    """
    Splits MNIST into 5 subsets (clients):
      - Client 1: digits {0,1}
      - Client 2: digits {2,3}
      - Client 3: digits {4,5}
      - Client 4: digits {6,7}
      - Client 5: digits {8,9}
    """
    digit_groups = [
        {0, 1},  # client 1
        {2, 3},  # client 2
        {4, 5},  # client 3
        {6, 7},  # client 4
        {8, 9},  # client 5
    ]

    all_subsets = []
    for group_id, dg in enumerate(digit_groups):
        indices = [i for i, (_, y) in enumerate(dataset) if y in dg]

        if case == "balanced":
            limit = 10000
        elif case == "imbalanced":
            # client 1 => 10k, others => 1k
            limit = 10000 if group_id == 0 else 1000
        elif case == "extremely_imbalanced":
            limit = 10000 if group_id == 0 else 100
        else:
            raise ValueError("Invalid case.")
        
        all_subsets.append(Subset(dataset, indices[:limit]))
    return all_subsets


def create_data_splits_k10(dataset, case):
    """
    Splits MNIST into 10 subsets (clients), each for one digit:
      - Client 1: digit {0}
      - Client 2: digit {1}
      - ...
      - Client 10: digit {9}
    """
    all_subsets = []
    for digit in range(10):
        indices = [i for i, (_, y) in enumerate(dataset) if y == digit]

        if case == "balanced":
            limit = 10000
        elif case == "imbalanced":
            # client 1 => 10k, others => 1k
            limit = 10000 if digit == 0 else 1000
        elif case == "extremely_imbalanced":
            limit = 10000 if digit == 0 else 100
        else:
            raise ValueError("Invalid case.")

        all_subsets.append(Subset(dataset, indices[:limit]))
    return all_subsets


# -------------------
#     Generator
# -------------------
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
            nn.ConvTranspose2d(64, 1, 5, stride=2, padding=2, output_padding=1),
            nn.Tanh()  # final tanh to push output into [-1,1]
        )

    def forward(self, x):
        return self.main(x)


# -------------------
#    Discriminator
# -------------------
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
            nn.Linear(7 * 7 * 128, 1),
            nn.Sigmoid()  # output in [0,1]
        )

    def forward(self, x):
        return self.main(x)


# ------------------------------
#  Generate & Save Images
# ------------------------------
def generate_and_save_images(generator, epoch, noise, folder="generated_images"):
    """
    Generates and saves images from the generator.
    """
    os.makedirs(folder, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        fake_images = generator(noise).cpu()
        fake_images = fake_images.view(fake_images.size(0), 28, 28)

        fig = plt.figure(figsize=(4, 4))
        for i in range(fake_images.size(0)):
            plt.subplot(4, 4, i + 1)
            plt.imshow(fake_images[i], cmap='gray')
            plt.axis('off')

        plt.savefig(f"{folder}/image_at_epoch_{epoch+1:04d}.png")
        plt.close()


# --------------------------------------------
#      FL-GAN (Federated Learning of a GAN)
# --------------------------------------------
class FLGAN:
    def __init__(self, noise_dim, case="balanced", num_clients=2, epochs=20):
        """
        Args:
            noise_dim: Latent dimension for the generator
            case: "balanced", "imbalanced", or "extremely_imbalanced"
            num_clients: number of clients (2, 5, or 10)
            epochs: total training epochs
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise_dim = noise_dim
        self.num_clients = num_clients
        self.epochs = epochs

        # Create local generators and discriminators
        self.generators = [Generator(noise_dim).to(self.device) for _ in range(num_clients)]
        self.discriminators = [Discriminator().to(self.device) for _ in range(num_clients)]

        # Create local optimizers
        self.optimizers_gen = [
            optim.Adam(g.parameters(), lr=0.0002, betas=(0.5, 0.999)) 
            for g in self.generators
        ]
        self.optimizers_disc = [
            optim.Adam(d.parameters(), lr=0.0002, betas=(0.5, 0.999))
            for d in self.discriminators
        ]

        # Prepare MNIST dataset + transform
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

        # Split dataset among clients
        if num_clients == 2:
            subsets = create_data_splits_k2(mnist_dataset, case)
        elif num_clients == 5:
            subsets = create_data_splits_k5(mnist_dataset, case)
        elif num_clients == 10:
            subsets = create_data_splits_k10(mnist_dataset, case)
        else:
            raise NotImplementedError("FL-GAN example supports num_clients in {2, 5, 10}.")

        # Build DataLoaders
        self.dataloaders = [
            DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)
            for ds in subsets
        ]

        self.criterion = nn.BCELoss()

    def train_discriminator(self, real_images, fake_images, discriminator, disc_optimizer):
        """
        Train one local discriminator on one batch: real_images, fake_images
        """
        disc_optimizer.zero_grad()

        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # Real loss
        outputs_real = discriminator(real_images)
        loss_real = self.criterion(outputs_real, real_labels)

        # Fake loss
        outputs_fake = discriminator(fake_images.detach())
        loss_fake = self.criterion(outputs_fake, fake_labels)

        disc_loss = loss_real + loss_fake
        disc_loss.backward()
        disc_optimizer.step()

        return disc_loss.item()

    def train_generator(self, generator, discriminator, gen_optimizer):
        """
        Train one local generator on one batch
        """
        gen_optimizer.zero_grad()

        noise = torch.randn(64, self.noise_dim, device=self.device)
        fake_images = generator(noise)

        # We want the discriminator to see these as real => label=1
        labels = torch.ones(fake_images.size(0), 1, device=self.device)
        outputs = discriminator(fake_images)
        gen_loss = self.criterion(outputs, labels)

        gen_loss.backward()
        gen_optimizer.step()

        return gen_loss.item()

    def train(self):
        # For visualization
        test_noise = torch.randn(16, self.noise_dim, device=self.device)

        generator_losses = []  # track average generator loss across all clients each epoch

        for epoch in range(self.epochs):
            print(f"\nEpoch [{epoch+1}/{self.epochs}]")
            epoch_gen_loss = 0.0

            # --- Train each client locally ---
            for client_id in range(self.num_clients):
                gen_i = self.generators[client_id]
                disc_i = self.discriminators[client_id]

                opt_g = self.optimizers_gen[client_id]
                opt_d = self.optimizers_disc[client_id]

                loader_i = self.dataloaders[client_id]

                running_gen_loss = 0.0
                n_batches = 0

                for real_images, _ in loader_i:
                    real_images = real_images.to(self.device)

                    # Create a batch of fake images
                    noise = torch.randn(real_images.size(0), self.noise_dim, device=self.device)
                    fake_images = gen_i(noise)

                    # 1) Train local discriminator
                    d_loss = self.train_discriminator(real_images, fake_images, disc_i, opt_d)

                    # 2) Train local generator
                    g_loss = self.train_generator(gen_i, disc_i, opt_g)

                    running_gen_loss += g_loss
                    n_batches += 1

                client_gen_loss = running_gen_loss / n_batches
                epoch_gen_loss += client_gen_loss
                print(f"  Client {client_id+1}: G-loss={client_gen_loss:.4f}, D-loss={d_loss:.4f}")

            # Average G-loss across all clients
            epoch_gen_loss /= self.num_clients
            generator_losses.append(epoch_gen_loss)

            print(f"  [Global] Epoch {epoch+1} Avg Generator Loss: {epoch_gen_loss:.4f}")

            # --- Aggregation Step (Global) ---
            # Average generator params across all clients => G_global
            global_gen_params = {}
            for key in self.generators[0].state_dict():
                # stack all clients' param tensors
                client_tensors = [g.state_dict()[key].float() for g in self.generators]
                global_gen_params[key] = torch.mean(torch.stack(client_tensors), dim=0)

            # Update each local generator with the global params
            for gen_i in self.generators:
                gen_i.load_state_dict(global_gen_params)

            # Save images from the "global" generator (arbitrarily pick client 0) every 10 epochs
            if (epoch + 1) % 10 == 0:
                global_gen = self.generators[0]
                generate_and_save_images(global_gen, epoch, test_noise)

        # Save generator losses to CSV
        with open("flgan_generator_losses.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "gen_loss"])
            for ep_idx, loss_val in enumerate(generator_losses, start=1):
                writer.writerow([ep_idx, loss_val])

        print("\nFL-GAN Training Complete.")
        print("Saved generator losses to flgan_generator_losses.csv")


# -------------------------
#       Main Script
# -------------------------
if __name__ == "__main__":
    # Example usage: FL-GAN with k=5 (clients), "imbalanced" case, 30 epochs
    fl_gan = FLGAN(
        noise_dim=128,
        case="balanced",  # or "imbalanced" / "extremely_imbalanced"
        num_clients=5,      # 2, 5, or 10
        epochs=200
    )
    fl_gan.train()
