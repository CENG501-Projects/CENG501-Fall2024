from model.model import CLEFModel, CLEFModelMultiLabel
from data_loader.data_loaders import EmoticDataset, CAER_SDataset
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import torch
from torchvision import transforms
from torch.optim import Adam
import torch.nn as nn
import torchvision.models as models
import os
from model.model import CLEFModel, CLEFModelMultiLabel
from model.emotic_model import Emotic
from model.caers_net_model import CAERSNet


from data_loader.data_loaders import EmoticDataset, CAER_SDataset
import argparse


from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm

def test_multi_label(model, device, test_loader):
    model.eval()
    map_metric = MultilabelAveragePrecision(num_labels=26).to(device)
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", unit="batch", dynamic_ncols=True)
        for idx, (full_images, face_images, context_images, labels) in enumerate(progress_bar):
            full_images, face_images, context_images, labels = (
                full_images.to(device),
                face_images.to(device),
                context_images.to(device),
                labels.to(device),
            )

            predictions, factual_fused, counterfactual_fused = model(full_images, face_images, context_images)
            # Update mAP metric with raw predictions and ground truth
            map_metric.update(predictions, labels.to(torch.int))

    map_score = map_metric.compute().item()  # Final mAP score

    return map_score


def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing the model.", unit="batch", dynamic_ncols=True)
        for idx, (full_images, face_images, context_images, labels) in enumerate(progress_bar):
            full_images ,face_images, context_images, labels = (
                full_images.to(device),
                face_images.to(device),
                context_images.to(device),
                labels.to(device),
            )
            predictions, factual_fused, counterfactual_fused = model(full_images, face_images, context_images)

            _, predicted = torch.max(predictions, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    return  accuracy




def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    parser = argparse.ArgumentParser()


    parser.add_argument("--test_data_path", help="The path to the test dataset", required=True,)
    parser.add_argument("--model_path", type=str, help="The path to the selected model to be tested.", required=True,)
    parser.add_argument("--num_classes", type=int, required = True)
    parser.add_argument("--model_type", help = "Either specify emotnet or caers", required="True")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model_path = "./model/resnet152_places365.pth"

    if args.model_type == "emotnet":
        model_path_places = './model/prepared_models'
        model_context = models.__dict__["resnet18"](num_classes=365)
        context_state_dict = torch.load(os.path.join(model_path_places, 'resnet18_state_dict.pth'))
        model_context.load_state_dict(context_state_dict)
        model_body = models.resnet18(pretrained=True)
        factual_branch = Emotic(list(model_context.children())[-1].in_features,
                                list(model_body.children())[-1].in_features,
                                nn.Sequential(*(list(model_context.children())[:-1])),
                                nn.Sequential(*(list(model_body.children())[:-1])))

        model = CLEFModelMultiLabel(factual_branch, num_classes=args.num_classes, resnet_model_path=resnet_model_path,).to(device)
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict)

        test_dataset = EmoticDataset("../data/emotic", args.test_data_path, transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        map =test_multi_label(model,device,test_loader)
        print("Mean average precision of the model:", map)
    elif args.model_type == "caers":

        test_coords_path = "coords/test_face_coords.json"
        test_dataset = CAER_SDataset(args.test_data_path, test_coords_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        # Initialize models
        factual_branch = CAERSNet()  # Factual branch from factual_temp.py
        model = CLEFModel(factual_branch, resnet_model_path, num_classes=args.num_classes).to(device)
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict)

        accuracy = test(model ,device, test_loader)
        print("Accuracy of the model:", accuracy)
    else:
        print("Either specify emotnet or caers as model_type!")

if __name__ == "__main__":
    main()