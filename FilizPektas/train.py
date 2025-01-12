from models import *
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn.functional as F
import copy
import pickle
from models import *
from utils import *
from MOLERE_algorithm import *
import os 

def train_erm_model(dataset_name, batch_size, results, device, input_size, epochs=100, use_existing_erm=False):
    if use_existing_erm and (os.path.exists(f"{dataset_name}_erm.pth") or os.path.exists(f"erm_{dataset_name}.pth")):
        dataset_train, dataset_test, delta = get_datasets(dataset_name, device, input_size)
    else:
        dataset_train, dataset_test, erm_model, delta = get_datasets(dataset_name, device, input_size)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Training ERM on {dataset_name}...")
    if use_existing_erm and (os.path.exists(f"{dataset_name}_erm.pth") or os.path.exists(f"erm_{dataset_name}.pth")):
        erm_model = torch.load(f"{dataset_name}_erm.pth")
    else:
        train_data, val_data, base_margins = prepare_data("random", dataset_train, delta=delta, device=device)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        erm_accuracy = train(erm_model, train_loader, val_loader, device, epochs=epochs)
        print(f"ERM Test Accuracy on {dataset_name}: {erm_accuracy:.2f}%")

        torch.save(erm_model, f"{dataset_name}_erm.pth")

    # Evaluate
    test_accuracy = evaluate(erm_model, test_loader, device)
    results["erm"].append(test_accuracy)
    # if dataset_name=="cifar100":
    #     with open(f'erm_margins_{dataset_name}.pkl', 'wb') as file:
    #         pickle.dump(base_margins, file)
    
    # Save the list to a file
    with open(f"results_{dataset_name}.pkl", 'wb') as file:
        pickle.dump(results, file)


def LRW_Opt_Training(dataset_name, device, results, batch_size, input_size, use_existing_opt=False, epochs=100):
    dataset_train, dataset_test, delta = get_datasets(dataset_name=dataset_name, device=device, input_size=input_size)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    erm_model = torch.load(f"{dataset_name}_erm.pth")

    if use_existing_opt and (os.path.exists(f"{dataset_name}_opt_classifier.pth") or os.path.exists(f"_opt_classifier_{dataset_name}.pth")):        
        trained_classifier = torch.load(f"{dataset_name}_opt_classifier.pth")
    else:
        # Initialize WideResNet Backbone
        backbone = copy.deepcopy(erm_model)
        splitter_net = SplitterNetwork(backbone, dataset_name)
        meta_net = MetaNetwork(backbone, dataset_name)

        # Define classifier using the WideResNet backbone
        classifier = backbone


        # Optimizers
        optimizer_classifier = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        optimizer_meta = optim.SGD(meta_net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        optimizer_splitter = optim.SGD(splitter_net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        # Training
        trained_classifier = train_molere_algorithm(
        dataset_train,
        classifier.to(device),
        meta_net.to(device),
        splitter_net.to(device),
        optimizer_classifier,
        optimizer_meta,
        optimizer_splitter,
        device,
        epochs=epochs,
        delta=delta,
        q_updates=5,
        batch_size=batch_size,
        warm_up_epoch_limit =26
        )

        torch.save(trained_classifier, f"{dataset_name}_opt_classifier.pth")
        torch.save(splitter_net, f"{dataset_name}_opt_splitter.pth")
        torch.save(meta_net, f"{dataset_name}_opt_metaNet.pth")


    # Evaluate
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    opt_accuracy = evaluate_model(trained_classifier, test_loader, device)

    # Save the list to a file
    with open(f'results_{dataset_name}.pkl', 'wb') as file:
        pickle.dump(results, file)
         
    # Train LRW variants
    results["LRW-Opt"].append(opt_accuracy)

def train_twice_training(dataset_name, device, results, batch_size, input_size, use_existing_hard=False, use_existing_easy=False, use_existing_random=False, epochs=100):
    dataset_train, dataset_test, delta = get_datasets(dataset_name=dataset_name, device=device, use_existing_erm=True, input_size=input_size)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    erm_model = torch.load(f"{dataset_name}_erm.pth")
    
    nc = len(dataset_train.classes)
    train_data, val_data, _ = prepare_data("random", dataset_train, delta=delta, device=device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    for mode in ["LRW-Hard", "LRW-Easy", "LRW-Random"]:
        try:
            if (mode=="LRW-Hard" and use_existing_hard and (os.path.exists(f"{dataset_name}_LRW-Hard.pth") or os.path.exists(f"LRW-Hard_{dataset_name}.pth"))) or \
                (mode=="LRW-Easy" and use_existing_hard and (os.path.exists(f"{dataset_name}_LRW-Easy.pth") or os.path.exists(f"LRW-Easy{dataset_name}.pth")))   or \
                (mode=="LRW-Random" and use_existing_hard and (os.path.exists(f"{dataset_name}_LRW-Random.pth") or os.path.exists(f"LRW-Random_{dataset_name}.pth"))):
                model = torch.load(f"{dataset_name}_{mode}.pth")
            else:
                print(f"Training LRW-{mode.capitalize()} on {dataset_name}...")
                train_data, val_data, margins = prepare_data(mode, dataset_train, delta=delta, erm_model=erm_model, device=device)
                
                model = get_dataset_model(dataset_name, nc, device)
                accuracy = train(model, train_loader, val_loader, device, epochs=epochs)
                torch.save(model, f"{dataset_name}_{mode}.pth")

            # if dataset_name=="cifar100":
            #     _, _, margins = prepare_data(mode, dataset_test, delta=delta, erm_model=erm_model, device=device)
            #     if mode == "LRW-Hard":
            #         with open(f'hard_margins_{dataset_name}.pkl', 'wb') as file:
            #             pickle.dump(margins, file)
            #     elif mode == "LRW-Easy":                
            #         with open(f'easy_margins_{dataset_name}.pkl', 'wb') as file:
            #             pickle.dump(margins, file)
                        
            
            # Save the list to a file
            with open(f'results_{dataset_name}.pkl', 'wb') as file:
                pickle.dump(results, file)

            # Evaluate
            test_accuracy = evaluate(model, test_loader, device)
            results[mode].append(test_accuracy)
            print(f"{mode} Test Accuracy on {dataset_name}: {test_accuracy:.2f}%")            
            # Save the list to a file
            with open(f"results_{dataset_name}.pkl", 'wb') as file:
                pickle.dump(results, file)
                
        except Exception as e:
            print("Training failed for LRW-", mode, " on ", dataset_name, " due to ", e)
            continue
