import torch
from torchvision import transforms
from torch.optim import Adam
from model.model import CLEFModel
from model.loss import CombinedLoss
from model.caers_net_model import CAERSNet  # Import the factual branch
from trainer.trainer import Trainer
from data_loader.data_loaders import CAER_SDataset


def main():
    # Hyperparameters
    train_data_path = "/data/Workspace/ybkaratas/caers/data/train"
    train_coord_path = "/data/Workspace/ybkaratas/caers/train_face_coords.json"
    test_coord_path = "/data/Workspace/ybkaratas/caers/test_face_coords.json"
    test_data_path = "/data/Workspace/ybkaratas/caers/data/test"
    num_classes = 7
    lambda_weight = 0.1
    learning_rate = 5e-5
    batch_size = 32
    num_epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_dataset = CAER_SDataset(train_data_path, train_coord_path,transform=transform)
    val_dataset = CAER_SDataset(test_data_path, test_coord_path,transform=transform)

    # Initialize models
    factual_branch = CAERSNet()  # Factual branch from factual_temp.py
    model = CLEFModel(factual_branch, num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = CombinedLoss(lambda_weight=lambda_weight)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Trainer setup
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        device=device,
        save_dir= "/data/Workspace/ybkaratas/trained_model"
    )

    # Start training
    trainer.train(num_epochs=num_epochs)


if __name__ == "__main__":
    main()
