import argparse
import torch
from utils import *
from train import *
import os
from visualizeResults import * 

def parse_args():
    parser = argparse.ArgumentParser(description="Run the model")
    parser.add_argument("--id-datasets", type=str, help="in-distribution experiment dataset names", default = "CatsDogs Airplane Cars cifar100 DR")
    parser.add_argument("--out-datasets", type=str, help="out-of-distribution experiment dataset names", default="")
    parser.add_argument("--use-id-pickle-file", type=bool, help="If exists, use existing in-distribution result file", default=False)
    parser.add_argument("--use-ood-pickle-file", type=bool, help="If exists, use existing out-of-distribution result file", default=False)
    parser.add_argument("--use-existing-erm", type=bool, help="If exists, use existing ERM model", default=False)
    parser.add_argument("--use-existing-opt", type=bool, help="If exists, use existing LRW-Opt model", default=False)
    parser.add_argument("--use-existing-hard", type=bool, help="If exists, use existing LRW-Hard model", default=False)
    parser.add_argument("--use-existing-easy", type=bool, help="If exists, use existing LRW-Easy model", default=False)
    parser.add_argument("--use-existing-random", type=bool, help="If exists, use existing LRW-Random model", default=False)
    parser.add_argument("--batch-size", type=int, help="Batch size", default=64)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=100)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Device
    inner_datasets = parse_datasets(args.id_datasets)
    outer_datasets = parse_datasets(args.out_datasets)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    epochs = args.epochs
    # Get a list of all files in the current directory
    files = os.listdir('.')
    if inner_datasets != []:
        print("Traversing inner datasets...")
        for dataset_name in inner_datasets: 
            print(f"Processing {dataset_name}...")    
            dataset_name = dataset_name.lower()   
            input_size = 224 if dataset_name == "cifar100" else 32
            results_id = get_results(args.use_id_pickle_file, dataset_name=dataset_name, files=files)    

            print(results_id)
            
            try:
                train_erm_model(dataset_name=dataset_name, batch_size=batch_size, results=results_id, device=device, input_size = input_size, epochs=epochs, use_existing_erm=args.use_existing_erm) 
            except Exception as e:     
                print(f"ERM Training failed due to {e}, skipping dataset...")
                continue

            try:
                LRW_Opt_Training(dataset_name=dataset_name, device=device, results=results_id, batch_size=batch_size, input_size=input_size, use_existing_opt=args.use_existing_opt, epochs=epochs)
            except Exception as e:
                print("LRW-Opt Training failed due to ", e)

            train_twice_training(dataset_name=dataset_name, device=device, results=results_id, batch_size=batch_size, input_size=input_size, epochs=epochs, use_existing_easy=args.use_existing_easy,
                                                                                                            use_existing_hard=args.use_existing_hard, use_existing_random=args.use_existing_random)
            
            if get_viz_check(results_id):
                visualize_results(dataset_name, results=results_id)


    print("Traversing outer datasets...")
    for dataset_name in outer_datasets:
        print(f"Processing {dataset_name}...")
        input_size = 224
        results_ood = get_results(args.use_ood_pickle_file, dataset_name=dataset_name, files=files) 
        
        try:
            train_erm_model(dataset_name=dataset_name, batch_size=batch_size, results=results_ood, device=device, input_size = input_size, epochs=epochs, use_existing_erm=args.use_existing_erm)
        except Exception as e:     
            print(f"ERM Training failed due to {e}, skipping dataset...")
            continue

        try:
            LRW_Opt_Training(dataset_name=dataset_name, device=device, results=results_ood, batch_size=batch_size, input_size=input_size, epochs=epochs, use_existing_opt=args.use_existing_opt)
        except Exception as e:
            print("LRW-Opt Training failed due to ", e)
        
        train_twice_training(dataset_name=dataset_name, device=device, results=results_ood, batch_size=batch_size, input_size=input_size, epochs=epochs, use_existing_easy=args.use_existing_easy,
                                                                                                            use_existing_hard=args.use_existing_hard, use_existing_random=args.use_existing_random)      
        
        if get_viz_check(results_ood):
            visualize_results(dataset_name, results=results_ood)
        
        if dataset_name == "cifar100" and get_margin_viz_check():
            cifar_hard_margins, cifar_easy_margins, erm_margins = get_margins(dataset_name, files)
            # ERM Margins are empty.
            if cifar_easy_margins != [] and cifar_hard_margins != []:
                try:
                    hard_margin_difference(erm_margins, cifar_hard_margins)
                except Exception as e:
                    print("Visualizing margin difference failed due to ", e)
                
                try:
                    bucket_difference(erm_margins, cifar_hard_margins, cifar_easy_margins)
                except Exception as e:
                    print("Visualizing margin difference failed due to ", e)
            else:
                print("Margins are empty.")
        
        elif not get_margin_viz_check():
            print("Models could not be found.")
