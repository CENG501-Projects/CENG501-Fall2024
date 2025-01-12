import torch.nn.functional as F
from torchvision.models import resnet152, resnet50
from wilds import get_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import FGVCAircraft, StanfordCars, CIFAR100, OxfordIIITPet
from torch.amp import autocast, GradScaler
import time
import numpy as np
from models import *
import os
from PIL import Image
from datasets import load_dataset, ReadInstruction
import pickle 

def parse_datasets(dataset_names):
    return dataset_names.split()

# Prepare data for training and validation
def prepare_data(mode, dataset, delta=0.1, model=None,  erm_model=None, device='cpu'):
    total_size = len(dataset)
    train_size = int((1 - delta) * total_size)
    val_size = total_size - train_size
    margins = []
    if mode in ["hard", "easy"]:
        if model is None:
            raise ValueError("Model is required for LRW mode.")
        data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
        margins = []
        with torch.no_grad():
            model.eval()
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = F.softmax(model(inputs), dim=1)
                correct_class_probs = outputs[range(labels.size(0)), labels]
                max_wrong_class_probs = outputs.clone()
                max_wrong_class_probs[range(labels.size(0)), labels] = 0
                max_wrong_class_probs = max_wrong_class_probs.max(dim=1)[0]
                margin = correct_class_probs - max_wrong_class_probs
                margins.extend(margin.cpu().numpy())
        margins = np.array(margins)
        indices = np.argsort(margins)

        print(f"Margins calculated. Mode: {mode}")
        print("Top 5 margins (easy examples):", margins[indices[:5]])
        print("Bottom 5 margins (hard examples):", margins[indices[-5:]])
        if mode == "easy":
            val_size = int(delta * len(indices))
            val_idx = indices[:val_size]
        elif mode == "hard":
            val_size = int(delta * len(indices))
            val_idx = indices[-val_size:]

        # Get training indices (those not in val_idx)
        train_idx = list(set(indices) - set(val_idx))
        train_data = Subset(dataset, train_idx)
        val_data = Subset(dataset, val_idx)
    elif mode == "erm":
        data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
        margins = []
        with torch.no_grad():
            erm_model.eval()
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = F.softmax(erm_model(inputs), dim=1)
                correct_class_probs = outputs[range(labels.size(0)), labels]
                max_wrong_class_probs = outputs.clone()
                max_wrong_class_probs[range(labels.size(0)), labels] = 0
                max_wrong_class_probs = max_wrong_class_probs.max(dim=1)[0]
                margin = correct_class_probs - max_wrong_class_probs
                margins.extend(margin.cpu().numpy())
        margins = np.array(margins)
        indices = np.argsort(margins)

        print(f"Margins calculated. Mode: {mode}")
        print("Top 5 margins (easy examples):", margins[indices[:5]])
        print("Bottom 5 margins (hard examples):", margins[indices[-5:]])

        train_data, val_data = random_split(dataset, [train_size, val_size])
    else:
        train_data, val_data = random_split(dataset, [train_size, val_size])

    print(f"Mode: {mode}, Train size: {len(train_data)}, Val size: {len(val_data)}")
    return train_data, val_data, margins



# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, *rest in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

# Training function
def train(model, train_loader, val_loader, device, epochs=100):
    # Move model to device
    model = model.to(device)

    # Optimizer, criterion, and scaler
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    for epoch in range(epochs):
        start_time = time.time()
        model.train()  # Set model to training mode

        running_loss = 0.0
        for i, (inputs, labels, *rest) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixed-precision training
            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backpropagation with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] completed in {epoch_time:.2f} seconds - Avg Loss: {avg_loss:.4f}")

        # Evaluate after each epoch
        val_accuracy = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

    return val_accuracy




def get_dataset_model(dataset_name, num_class, device):
    if dataset_name == "catsdogs":
        # Use ResNet-152 for these datasets
        model = resnet152(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_class)  # Replace 100 with the number of classes as required
    elif dataset_name == "cifar100":
        # Wide Resnet implementation is taken from 
        model = Wide_ResNet(28, 10, 0.3, 100)
    elif dataset_name == "cars" or dataset_name == "airplane":
        model = resnet32(num_classes=num_class, dropout_rate=0.3).to(device)
    else:
        model = resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, num_class)
    return model.to(device)

def get_transforms(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
LOCAL_DATASET_DIR = '/content/drive/MyDrive/diabetic_retinopathy'
def download_and_prepare_dataset():
    if not os.path.exists(LOCAL_DATASET_DIR):
        os.makedirs(LOCAL_DATASET_DIR)
        print("Downloading Diabetic Retinopathy Dataset...")
        ri = ReadInstruction('train') + ReadInstruction('validation')
        dataset_dr = load_dataset('youssefedweqd/Diabetic_Retinopathy_Detection_preprocessed2', split=ri)
        print("Saving dataset locally...")

        features, labels = [], []
        for i, example in enumerate(dataset_dr):
            img_path = os.path.join(LOCAL_DATASET_DIR, f"img_{i}.jpg")
            example['image'].save(img_path, format='JPEG')
            features.append(img_path)
            labels.append(example['label'])

        # Save metadata
        np.save(os.path.join(LOCAL_DATASET_DIR, "features.npy"), features)
        np.save(os.path.join(LOCAL_DATASET_DIR, "labels.npy"), labels)
        print("Dataset saved locally.")
    else:
        print("Dataset already exists locally.")

    features = np.load(os.path.join(LOCAL_DATASET_DIR, "features.npy"), allow_pickle=True)
    labels = np.load(os.path.join(LOCAL_DATASET_DIR, "labels.npy"), allow_pickle=True)
    return features, labels
    
class DiabeticRetinopathy(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = Image.open(self.features[idx]).convert('RGB')
        if self.transform:
            feature = self.transform(feature)
        label = self.labels[idx]
        return feature, label

def get_datasets(dataset_name, device, use_existing_erm=False, input_size=224):
    
    transform = get_transforms(input_size)

    if dataset_name == "catsdogs":
        dataset_train = OxfordIIITPet(root="./data", split="trainval", download=True, transform=transform)
        dataset_test = OxfordIIITPet(root="./data", split="test", download=True, transform=transform)
        erm_model = get_dataset_model(dataset_name, len(dataset_train.classes), device)
        delta = 0.163043478

    elif dataset_name == "cars":
        # Cars dataset is not available in the location provided by torchvision's download command, so we need to download it from Kaggle
        import kaggle # Kaggle requires an API key, you need to import that.
        kaggle.api.dataset_download_files('rickyyyyyyy/torchvision-stanford-cars', path='./data', unzip=True)
        dataset_train = StanfordCars(root="./data", split="train", download=False, transform=transform)
        dataset_test = StanfordCars(root="./data", split="test", download=False, transform=transform)
        erm_model =  get_dataset_model(dataset_name, len(dataset_train.classes), device)
        delta = 1 / 16

    elif dataset_name == "cifar100":
        dataset_train = CIFAR100(root="./data", train=True, download=True, transform=transform)
        dataset_test = CIFAR100(root="./data", train=False, download=True, transform=transform)
        erm_model = get_dataset_model(dataset_name, len(dataset_train.classes), device)
        delta = 0.1

    elif dataset_name == "airplane":
        dataset_train = FGVCAircraft(
            root='./data',
            split='trainval',
            transform=transform,
            download=True
        )

        dataset_test = FGVCAircraft(
            root='./data',
            split='test',
            transform=transform,
            download=True
        )
        erm_model = get_dataset_model(dataset_name, len(dataset_train.classes), device)
        delta = 0.1

    elif dataset_name == "dr":

        features, labels = download_and_prepare_dataset()
        dataset = DiabeticRetinopathy(features, labels, transform=get_transforms(input_size))

        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        dataset_train, dataset_test = random_split(dataset, [train_size, test_size])

        erm_model = get_dataset_model(dataset_name, len(dataset_train.classes), device)
        delta = 0.1
        
    elif dataset_name == "camleyon":
      dataset_camleyon = get_dataset(dataset="camelyon17", download=True)
      train_data_camleyon = dataset_camleyon.get_subset(
          "train",
          transform=transforms.Compose(
              [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
          ),
      )

      val_data_camleyon = dataset_camleyon.get_subset(
          "val",
          transform=transforms.Compose(
              [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
          ),
      )

      dataset_test = dataset_camleyon.get_subset(
          "test",
          transform=transforms.Compose(
              [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
          ),
      )

      # Concatenate the datasets
      dataset_train = ConcatDataset([train_data_camleyon, val_data_camleyon])

      erm_model = get_dataset_model(dataset_name, train_data_camleyon._n_classes, device)
      delta = 0.1
      return train_data_camleyon, val_data_camleyon, dataset_test, erm_model, delta

    elif dataset_name == "iwildcam":
      # Load the full dataset, and download it if necessary
      dataset_iwildCam = get_dataset(dataset="iwildcam", download=True)

      # Get the training set
      train_data_iwildCam = dataset_iwildCam.get_subset(
          "train",
          transform=transforms.Compose(
              [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
          ),
      )

      val_data_iwildCam = dataset_iwildCam.get_subset(
          "val",
          transform=transforms.Compose(
              [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
          ),
      )

      # Concatenate the datasets
      dataset_train = ConcatDataset([train_data_iwildCam, val_data_iwildCam])

      dataset_test = dataset_iwildCam.get_subset(
          "test",
          transform=transforms.Compose(
              [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
          ),
      )
      erm_model = get_dataset_model(dataset_name, train_data_iwildCam._n_classes, device)
      delta = 0.1
      return train_data_iwildCam, val_data_iwildCam, dataset_test, erm_model, delta
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    
    if use_existing_erm and (os.path.exists(f"{dataset_name}_erm.pth") or os.path.exists(f"erm_{dataset_name}.pth")):
        return dataset_train, dataset_test, delta
    else:
        return dataset_train, dataset_test, erm_model, delta

def get_results(use_pickle_file, dataset_name, files):
    if use_pickle_file:
        for file in files:
            if file.endswith('.pkl') and 'results' in file and dataset_name in file:
                print(f"Loading file: {file}")
                # Load the pickle file
                with open(file, 'rb') as f:
                    results = pickle.load(f)
                # Print or process the loaded data
                print(f"Data from {file}: {results}")
    else:
        results = {
            "erm": [],
            "LRW-Opt": [],
            "LRW-Hard": [],
            "LRW-Easy": [],
            "LRW-Random": []
            }
        
    return results

def get_margin_viz_check(dataset_name):
    # check if Hard and Easy models exist
    return (os.path.exists(f"{dataset_name}_LRW-Hard.pth") or os.path.exists(f"LRW-Hard_{dataset_name}.pth")) and \
    (os.path.exists(f"{dataset_name}_LRW-Easy.pth") or os.path.exists(f"LRW-Easy{dataset_name}.pth"))

def get_margins(dataset_name, device):
    _, dataset_test, erm_model, delta = get_datasets(dataset_name, device)
    hard_model = torch.load(f"{dataset_name}_LRW-Hard.pth")
    easy_model = torch.load(f"{dataset_name}_LRW-Easy.pth")
    erm_model = torch.load(f"{dataset_name}_erm.pth")

    # ERM Mode is only used to acquire ERM margins
    _, _, base_margins = prepare_data("erm", erm_model, dataset_test, delta=delta, erm_model=erm_model, device=device)
    _, _, cifar_hard_margins = prepare_data("easy", easy_model, dataset_test, delta=delta, erm_model=erm_model, device=device)
    _, _, cifar_easy_margins = prepare_data("hard", hard_model, dataset_test, delta=delta, erm_model=erm_model, device=device)
    return cifar_hard_margins, cifar_easy_margins, base_margins

def get_viz_check(results):
    # We need ERM and one another method to visualize
    if results["erm"] != [] and (results["LRW-Opt"] != [] or results["LRW-Hard"] != [] or results["LRW-Easy"] != [] or results["LRW-Random"] != []):
        return True
    return False

def calculate_sem(data):
    """
    Calculate the Standard Error of the Mean (SEM) for a given list of data points.

    Parameters:
        data (list or numpy array): A list or array of numerical values.

    Returns:
        float: The Standard Error of the Mean (SEM).
    """
    if len(data) == 0:
        raise ValueError("The data list cannot be empty.")

    std_dev = np.std(data, ddof=1)  # Calculate standard deviation (ddof=1 for sample standard deviation)
    sample_size = len(data)
    sem = std_dev / np.sqrt(sample_size)  # Calculate SEM
    return sem
