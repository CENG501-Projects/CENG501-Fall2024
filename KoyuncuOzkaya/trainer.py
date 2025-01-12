import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def train(
    encoder,                # HierarchicalEncoder
    decoder,                # PointDecoder
    train_dataloader,       # DataLoader for training
    val_dataloader,         # DataLoader for validation
    device='cuda',
    num_epochs=5,
    checkpoint_dir='models',  # Directory to save checkpoints
    patience=10                     # Early stopping patience
):

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    encoder.to(device)
    decoder.to(device)

    # Combine the parameters for both encoder & decoder
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-4)

    # Initialize loss history
    loss_history = {
        'train_total': [],
        'train_occ': [],
        'train_col': [],
        'val_total': [],
        'val_occ': [],
        'val_col': []
    }
    epochs = []
    
    # Initialize Matplotlib for real-time plotting
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.grid(True)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()
        running_loss = 0.0
        running_loss_occ = 0.0
        running_loss_col = 0.0

        for it, batch in enumerate(train_dataloader):
            p_world = batch['p_world'].to(device)    # [B,3]
            occ_gt  = batch['occ_gt'].to(device)     # [B,1]
            col_gt  = batch['color_gt'].to(device)   # [B,3]
            p_coords = torch.cat([occ_gt, col_gt], dim=1)
            # 1) Encode the point => phi_combined
            phi_combined = encoder(p_coords,p_world)  # shape [B, feat_dim]
            # 2) Decode => occupancy + color
            occ_pred, col_pred = decoder(phi_combined, p_world)
            # occ_pred => [B, 1], col_pred => [B, 3]

            # 3) Compute losses
            loss_occ = F.mse_loss(occ_pred, occ_gt)  # or BCE if occupancy is in [0,1]
            loss_col = F.mse_loss(col_pred, col_gt)
            loss = loss_occ + loss_col

            # 4) Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses
            running_loss += loss.item()
            running_loss_occ += loss_occ.item()
            running_loss_col += loss_col.item()

            if it % 50 == 0:
                print(f"[Epoch {epoch}, Iter {it}] loss={loss.item():.4f} (occ={loss_occ.item():.4f}, col={loss_col.item():.4f})")

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_dataloader)
        avg_train_loss_occ = running_loss_occ / len(train_dataloader)
        avg_train_loss_col = running_loss_col / len(train_dataloader)

        loss_history['train_total'].append(avg_train_loss)
        loss_history['train_occ'].append(avg_train_loss_occ)
        loss_history['train_col'].append(avg_train_loss_col)

        # Validation phase
        encoder.eval()
        decoder.eval()
        val_running_loss = 0.0
        val_running_loss_occ = 0.0
        val_running_loss_col = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                p_world = batch['p_world'].to(device)    # [B,3]
                occ_gt  = batch['occ_gt'].to(device)     # [B,1]
                col_gt  = batch['color_gt'].to(device)   # [B,3]
                p_coords = torch.cat([occ_gt, col_gt], dim=1)
                phi_combined = encoder(p_coords, p_world)
                occ_pred, col_pred = decoder(phi_combined, p_world)

                loss_occ = F.mse_loss(occ_pred, occ_gt)
                loss_col = F.mse_loss(col_pred, col_gt)
                loss = loss_occ + loss_col

                val_running_loss += loss.item()
                val_running_loss_occ += loss_occ.item()
                val_running_loss_col += loss_col.item()

        avg_val_loss = val_running_loss / len(val_dataloader)
        avg_val_loss_occ = val_running_loss_occ / len(val_dataloader)
        avg_val_loss_col = val_running_loss_col / len(val_dataloader)

        loss_history['val_total'].append(avg_val_loss)
        loss_history['val_occ'].append(avg_val_loss_occ)
        loss_history['val_col'].append(avg_val_loss_col)

        epochs.append(epoch)

        print(f"==> Epoch {epoch} Summary:")
        print(f"    Train Loss: {avg_train_loss:.4f} (occ: {avg_train_loss_occ:.4f}, col: {avg_train_loss_col:.4f})")
        print(f"    Val   Loss: {avg_val_loss:.4f} (occ: {avg_val_loss_occ:.4f}, col: {avg_val_loss_col:.4f})")

        # Update the plot
        ax.clear()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.grid(True)
        ax.plot(epochs, loss_history['train_total'], label='Train Total Loss', color='blue')
        ax.plot(epochs, loss_history['val_total'], label='Val Total Loss', color='orange')
        ax.plot(epochs, loss_history['train_occ'], label='Train Occupancy Loss', color='green', linestyle='--')
        ax.plot(epochs, loss_history['val_occ'], label='Val Occupancy Loss', color='red', linestyle='--')
        ax.plot(epochs, loss_history['train_col'], label='Train Color Loss', color='purple', linestyle=':')
        ax.plot(epochs, loss_history['val_col'], label='Val Color Loss', color='brown', linestyle=':')
        ax.legend()
        plt.show()
        plt.pause(0.1)  # Brief pause to update the plot

        # Checkpointing based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            checkpoint_path = os.path.join(checkpoint_dir, f'best_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, checkpoint_path)
            print(f"    [Checkpoint] Saved best model at epoch {epoch} with val loss {avg_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"    [No Improvement] {epochs_no_improve} epochs without improvement.")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    plt.ioff()
    
    print("Training completed!")

    # Optionally, save the final model
    final_checkpoint = os.path.join(checkpoint_dir, f'final_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
    }, final_checkpoint)
    print(f"Final model saved at epoch {epoch}.")
