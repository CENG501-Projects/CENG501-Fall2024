import torch
from torchvision import transforms
from torch.optim import Adam
import torch.nn as nn
import torchvision.models as models
import os
from model.model import CLEFModel, CLEFModelMultiLabel
from model.loss import CombinedLoss, CombinedLossMultiLabel
from model.emotic_model import Emotic
from model.caers_net_model import CAERSNet
from trainer.trainer import Trainer, TrainerMultiLabel
from data_loader.data_loaders import EmoticDataset, CAER_SDataset
import argparse


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    parser = argparse.ArgumentParser()


    parser.add_argument("--train_data_path", help="The path to the train dataset", default="../data/emotic_pre/train.csv")
    parser.add_argument("--validation_data_path", help="The path to the validation dataset", default="../data/emotic_pre/val.csv")
    parser.add_argument("--factual_model", type=str, help="The selection of the factual model.", default="emotnet")
    parser.add_argument("--num_classes", type=int, default=26)
    parser.add_argument("--lr", type=int, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=75)


    # Parse the arguments
    args = parser.parse_args()

    train_data_path = args.train_data_path #
    test_data_path = args.validation_data_path #
    num_classes = args.num_classes
    learning_rate = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resnet_model_path = "./model/resnet152_places365.pth"

    if args.factual_model == "emotnet":
        print("EMOTNET Selected, default paths and values will be used.")
        root_dir = "../data/emotic"
        train_dataset = EmoticDataset(root_dir, train_data_path, transform)
        val_dataset = EmoticDataset(root_dir, test_data_path, transform)
        model_path_places = './model/prepared_models'
        model_context = models.__dict__["resnet18"](num_classes=365)
        context_state_dict = torch.load(os.path.join(model_path_places, 'resnet18_state_dict.pth'))
        model_context.load_state_dict(context_state_dict)
        model_body = models.resnet18(pretrained=True)
        factual_branch = Emotic(list(model_context.children())[-1].in_features,
                                list(model_body.children())[-1].in_features,
                                nn.Sequential(*(list(model_context.children())[:-1])),
                                nn.Sequential(*(list(model_body.children())[:-1])))

        criterion = CombinedLossMultiLabel()
        model = CLEFModelMultiLabel(factual_branch, num_classes=num_classes, resnet_model_path=resnet_model_path).to(
            device)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        trainer = TrainerMultiLabel(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            device=device,
            save_dir="./output"
        )
    elif args.factual_model == "caers":
        train_coords_path = "coords/train_face_coords.json"
        test_coords_path = "coords/test_face_coords.json"
        train_dataset = CAER_SDataset(train_data_path, train_coords_path, transform=transform)
        val_dataset = CAER_SDataset(test_data_path, test_coords_path, transform=transform)

        # Initialize models
        factual_branch = CAERSNet()  # Factual branch from factual_temp.py
        model = CLEFModel(factual_branch, resnet_model_path, num_classes=num_classes).to(device)

        # Loss and optimizer
        criterion = CombinedLoss()
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
            save_dir="./output"
        )
    else:
        print("Either select caers or emotnet as the factual branch!")


    trainer.train(num_epochs=num_epochs)

if __name__ == "__main__":
    main()







