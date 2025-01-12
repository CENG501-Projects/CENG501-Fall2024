import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import os
import csv

# ------------------
#     Generator
# ------------------
class Generator(nn.Module):
    def __init__(self, noise_dim=128):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(noise_dim, 7 * 7 * 256),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
            # Notice: no final Sigmoid (we use BCEWithLogitsLoss)
        )

    def forward(self, x):
        return self.main(x)


# ------------------
#    Discriminator
# ------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(7 * 7 * 128, 1)
            # No Sigmoid => We'll feed raw logits to BCEWithLogitsLoss.
        )

    def forward(self, x):
        return self.main(x)


# ----------------------------------
#   Data Split Helpers for K = 2
# ----------------------------------
def create_data_splits_k2(dataset, case="balanced"):
    """
    Splits MNIST into two subsets:
      Client 1: Digits {0,1,2,3,4}
      Client 2: Digits {5,6,7,8,9}
    """
    c1_indices = [i for i, (img, label) in enumerate(dataset) if label in {0,1,2,3,4}]
    c2_indices = [i for i, (img, label) in enumerate(dataset) if label in {5,6,7,8,9}]

    if case == "balanced":
        c1_indices = c1_indices[:10000]
        c2_indices = c2_indices[:10000]
    elif case == "imbalanced":
        c1_indices = c1_indices[:10000]
        c2_indices = c2_indices[:1000]
    elif case == "extremely_imbalanced":
        c1_indices = c1_indices[:10000]
        c2_indices = c2_indices[:100]
    else:
        raise ValueError("Invalid case. Choose from 'balanced', 'imbalanced', or 'extremely_imbalanced'.")

    c1_data = Subset(dataset, c1_indices)
    c2_data = Subset(dataset, c2_indices)
    return [c1_data, c2_data]


# ----------------------------------
#   Data Split Helpers for K = 5
# ----------------------------------
def create_data_splits_k5(dataset, case="balanced"):
    """
    Splits MNIST into five subsets:
      Client 1: digits {0,1}
      Client 2: digits {2,3}
      Client 3: digits {4,5}
      Client 4: digits {6,7}
      Client 5: digits {8,9}
    """

    digit_groups = [
        {0,1},  # client1
        {2,3},  # client2
        {4,5},  # client3
        {6,7},  # client4
        {8,9},  # client5
    ]

    # Gather indices
    all_indices = []
    for dg in digit_groups:
        idx = [i for i, (img, label) in enumerate(dataset) if label in dg]
        all_indices.append(idx)

    def slice_limit(case, client_id):
        if case == "balanced":
            return 10000
        elif case == "imbalanced":
            # client 1 => 10k, others => 1k
            return 10000 if client_id == 0 else 1000
        elif case == "extremely_imbalanced":
            return 10000 if client_id == 0 else 100
        else:
            raise ValueError("Invalid case.")

    subsets = []
    for i, idx_list in enumerate(all_indices):
        limit = slice_limit(case, i)
        subsets.append(Subset(dataset, idx_list[:limit]))

    return subsets


# -----------------------------------
#   Data Split Helpers for K = 10
# -----------------------------------
def create_data_splits_k10(dataset, case="balanced"):
    """
    Splits MNIST into 10 subsets, each having exactly one digit:
      Client 1: digit {0}
      Client 2: digit {1}
      ...
      Client 10: digit {9}
    """

    digit_groups = [{d} for d in range(10)]  # 0..9
    all_indices = []
    for dg in digit_groups:
        idx = [i for i, (img, label) in enumerate(dataset) if label in dg]
        all_indices.append(idx)

    def slice_limit(case, client_id):
        if case == "balanced":
            return 10000
        elif case == "imbalanced":
            # client 1 => 10k, others => 1k
            return 10000 if client_id == 0 else 1000
        elif case == "extremely_imbalanced":
            return 10000 if client_id == 0 else 100
        else:
            raise ValueError("Invalid case.")

    subsets = []
    for i, idx_list in enumerate(all_indices):
        limit = slice_limit(case, i)
        subsets.append(Subset(dataset, idx_list[:limit]))

    return subsets


# ---------------------------
#   Generate & Save Images
# ---------------------------
def generate_and_save_images(generator, epoch, noise, folder="mdgan_generated_images"):
    os.makedirs(folder, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        fake_images = generator(noise).cpu()
        # optional: clamp or tanh to push into [-1,1]
        fake_images = torch.tanh(fake_images)

        # reshape for plotting
        fake_images = fake_images.view(fake_images.size(0), 28, 28)

        fig = plt.figure(figsize=(4, 4))
        for i in range(fake_images.size(0)):
            plt.subplot(4, 4, i+1)
            plt.imshow(fake_images[i], cmap='gray')
            plt.axis('off')

        plt.savefig(f"{folder}/epoch_{epoch+1:03d}.png")
        plt.close()


# -----------------------------------
#   Training Routines (MD-GAN)
# -----------------------------------
def train_local_discriminator(
    discriminator, optimizer, real_images, generator, device, noise_dim
):
    """
    Train one local discriminator (on one batch).
    Returns the discriminator loss (float).
    """
    discriminator.train()
    optimizer.zero_grad()

    real_images = real_images.to(device)
    # Real label smoothing => 0.9
    real_labels = torch.full((real_images.size(0), 1), 0.9, device=device)

    real_logits = discriminator(real_images)
    real_loss = nn.BCEWithLogitsLoss()(real_logits, real_labels)

    # Fake forward pass
    noise = torch.randn(real_images.size(0), noise_dim, device=device)
    fake_images = generator(noise)
    fake_labels = torch.zeros(real_images.size(0), 1, device=device)  # fake => 0.0
    fake_logits = discriminator(fake_images.detach())
    fake_loss = nn.BCEWithLogitsLoss()(fake_logits, fake_labels)

    disc_loss = real_loss + fake_loss
    disc_loss.backward()
    optimizer.step()

    return disc_loss.item()


def accumulate_generator_gradients(
    generator, discriminators, device, noise_dim, batch_size=128
):
    """
    Aggregator step: sum (or average) generator's gradients from each local discriminator.
    We'll do one forward pass per discriminator, accumulate the gradient,
    then do one global optimizer step.
    """
    generator.train()
    # Zero out any old gradients from the aggregator
    for p in generator.parameters():
        p.grad = None

    total_gen_loss = 0.0
    # Accumulate gradient from each local discriminator
    for disc in discriminators:
        noise = torch.randn(batch_size, noise_dim, device=device)
        fake_images = generator(noise)
        # Generator wants disc output == 1.0 (fake as real)
        gen_labels = torch.ones(batch_size, 1, device=device)

        logits = disc(fake_images)
        gen_loss = nn.BCEWithLogitsLoss()(logits, gen_labels)

        # accumulate gradient
        gen_loss.backward(retain_graph=True)
        total_gen_loss += gen_loss.item()

    return total_gen_loss


# -----------------------
#   MD-GAN Main
# -----------------------
def md_gan_main(
    num_workers=2,    # i.e. number of clients
    noise_dim=128,
    epochs=20,
    batch_size=128,
    case="balanced"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"MD-GAN with k={num_workers}, case='{case}'")

    # 1. Create generator + local discriminators
    generator = Generator(noise_dim).to(device)
    discriminators = [Discriminator().to(device) for _ in range(num_workers)]

    # 2. Create optimizers
    g_lr = 0.0002
    d_lr = 0.0002
    beta1, beta2 = 0.5, 0.999
    gen_optimizer = optim.Adam(generator.parameters(), lr=g_lr, betas=(beta1, beta2))
    disc_optimizers = [optim.Adam(d.parameters(), lr=d_lr, betas=(beta1, beta2)) 
                       for d in discriminators]

    # 3. Load MNIST & split among k clients
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Based on k, pick the correct splitting function
    if num_workers == 2:
        worker_datasets = create_data_splits_k2(mnist_dataset, case)
    elif num_workers == 5:
        worker_datasets = create_data_splits_k5(mnist_dataset, case)
    elif num_workers == 10:
        worker_datasets = create_data_splits_k10(mnist_dataset, case)
    else:
        raise NotImplementedError(
            f"Only k=2, k=5, or k=10 are supported. Got k={num_workers}."
        )

    # Create data loaders for each client
    worker_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        for ds in worker_datasets
    ]

    # 4. Fixed noise for generating samples
    fixed_noise = torch.randn(16, noise_dim, device=device)
    generator_losses = []

    # 5. Main Training Loop
    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")

        # -- A) Train each local discriminator --
        disc_losses = [0.0] * num_workers
        for worker_id in range(num_workers):
            running_loss = 0.0
            for real_images, _ in worker_loaders[worker_id]:
                loss_d = train_local_discriminator(
                    discriminator=discriminators[worker_id],
                    optimizer=disc_optimizers[worker_id],
                    real_images=real_images,
                    generator=generator,
                    device=device,
                    noise_dim=noise_dim
                )
                running_loss += loss_d
            disc_losses[worker_id] = running_loss / len(worker_loaders[worker_id])

        # -- B) Aggregator updates the Generator once (accumulate grads) --
        gen_optimizer.zero_grad()
        total_gen_loss = accumulate_generator_gradients(
            generator=generator,
            discriminators=discriminators,
            device=device,
            noise_dim=noise_dim,
            batch_size=batch_size
        )
        # Single optimizer step for the global generator
        gen_optimizer.step()

        avg_gen_loss = total_gen_loss / num_workers
        generator_losses.append(avg_gen_loss)

        # Logging
        print(f"  Generator Loss: {avg_gen_loss:.4f}")
        for w, d_loss in enumerate(disc_losses):
            print(f"  D{w+1} Loss: {d_loss:.4f}")

        # -- C) Generate images periodically
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            generate_and_save_images(generator, epoch, fixed_noise)

    # 6. Save generator loss to CSV
    with open("mdgan_generator_losses.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "gen_loss"])
        for epoch_idx, loss_val in enumerate(generator_losses, start=1):
            writer.writerow([epoch_idx, loss_val])

    print("\nMD-GAN Training Complete.")
    # Plot generator loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs+1), generator_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MD-GAN Generator Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("mdgan_generator_loss.png")
    plt.show()
    print("All done!")


# -------------------
#   Run the Script
# -------------------
if __name__ == "__main__":
    md_gan_main(
        num_workers=5,          # can be 2, 5, or 10
        noise_dim=128,
        epochs=200,              # or more if needed
        batch_size=128,
        case="balanced"       # or "balanced" / "extremely_imbalanced"
    )
