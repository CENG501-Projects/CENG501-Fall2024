import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from cimd_utils import *
from two_branch_network import TwoBranchNetwork
import wandb
from tqdm import tqdm
import os

torch.manual_seed(501)
np.random.seed(501)

def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=100,
    learning_rate=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # check if best_checkpoint.pth exists
    checkpoint_path = 'best_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        start_epoch = 0
    
    wandb.init(
        project="cimd-training",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "device": device,
        }
    )
    best_test_loss = 1e9
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_f1 = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for batch_idx, (images, masks, class_labels) in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = criterion(outputs, masks)

            batch_acc = get_accuracy(outputs, class_labels)
            running_acc += batch_acc
            
            batch_f1 = calculate_pixel_f1(masks, outputs)
            running_f1 += batch_f1
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            
            wandb.log({
                "batch_loss": loss.item(),
                "batch_acc": batch_acc,
                "batch_f1": batch_f1
            })
            
            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}")
            progress_bar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        f1 = running_f1 / len(train_loader)
                
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_f1 = 0.0
        with torch.no_grad():
            for images, masks, class_labels in test_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                test_loss += criterion(outputs, masks).item()
                test_acc += get_accuracy(outputs, class_labels)
                test_f1 += calculate_pixel_f1(masks, outputs)
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        test_f1 /= len(test_loader)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "train_f1": f1,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1
        })
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Average Training Loss: {epoch_loss:.4f}")
        print(f"Average Test Loss: {test_loss:.4f}")
        
        # save model checkpoint if test loss has decreased
        if epoch == 0 or test_loss < best_test_loss:
            best_test_loss = test_loss
            checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss,
            'test_loss': test_loss
            }
            torch.save(checkpoint, 'best_checkpoint.pth')
            # wandb.save('best_checkpoint.pth')
                
    wandb.finish()
    return model

if __name__ == "__main__":
    input_paths, output_paths, class_labels = get_io_paths()
    dataset = ImageDataset(input_paths, output_paths, class_labels)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    c1, c2, c3, c4 = 18, 36, 72, 144
    model = TwoBranchNetwork(c1, c2, c3, c4)

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=100,
        learning_rate=0.001
    )