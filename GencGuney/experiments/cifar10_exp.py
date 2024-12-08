import os
import torch
from data.loaders import get_dataloaders
from model.zoo import ModelZoo
from model.scoring import msp_score, compute_score
from zode.algorithm import zode_algorithm
from zode.evaluation import evaluate_model
import numpy as np


def run_cifar10_experiment(data_dir="data", batch_size=64, alpha=0.05):
    """
    Run the CIFAR10-based experiment for OOD detection using the ZODE algorithm.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for DataLoaders.
        alpha (float): Target TPR level for the ZODE algorithm.

    Returns:
        None: Prints results for all OOD datasets.
    """
    print("### CIFAR10 OOD Detection Experiment ###")

    # Step 1: Load datasets
    print("Loading datasets...")
    dataloaders = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    cifar10_test_loader = dataloaders["cifar10_test"]

    ood_loaders = {
        "SVHN": dataloaders["svhn_test"],
        "LSUN": dataloaders["lsun_test"],
        "iSUN": dataloaders["isun_test"],
        "Places365": dataloaders["places365_test"],
        "Texture": dataloaders["texture_test"],
    }
    print("Datasets loaded successfully.\n")

    # Step 2: Initialize Model Zoo
    print("Initializing Model Zoo...")
    model_zoo = ModelZoo()
    model_zoo.initialize_zoo()
    print(f"Model Zoo contains: {model_zoo.get_model_names()}\n")

    # Step 3: Compute validation features for ZODE
    print("Extracting features for CIFAR10 validation set (in-distribution)...")
    cifar10_features = []
    cifar10_labels = []

    for batch, labels in cifar10_test_loader:
        batch = batch.to(model_zoo.device)
        for model_info in model_zoo.models:
            model = model_info["model"]
            features = model(batch).cpu()
            cifar10_features.append(features)
            cifar10_labels.append(labels)

    # Combine validation features
    cifar10_features = torch.cat(cifar10_features)
    cifar10_labels = torch.cat(cifar10_labels)
    print("Validation features extracted.\n")

    # Step 4: Evaluate on each OOD dataset
    results = {}
    for ood_name, ood_loader in ood_loaders.items():
        print(f"Evaluating on {ood_name} dataset...")
        test_scores = []
        test_labels = []

        for batch, labels in ood_loader:
            batch = batch.to(model_zoo.device)
            for model_info in model_zoo.models:
                model = model_info["model"]
                # Compute OOD scores using MSP for simplicity
                scores = [compute_score(sample, model, msp_score) for sample in batch]
                test_scores.extend(scores)

            # Assign OOD labels (0 for OOD samples)
            test_labels.extend([0] * len(batch))

        # Combine CIFAR10 and OOD scores for evaluation
        combined_scores = np.array(test_scores)
        combined_labels = np.array(test_labels)

        # Run ZODE on combined scores
        print("Running ZODE algorithm...")
        zode_results = zode_algorithm(combined_scores, model_zoo.models, cifar10_features, msp_score, alpha)

        # Evaluate performance
        metrics = evaluate_model(combined_scores, combined_labels, target_tpr=1 - alpha)
        results[ood_name] = metrics
        print(f"Results for {ood_name}: {metrics}\n")

    # Step 5: Summarize results
    print("### Experiment Summary ###")
    for ood_name, metrics in results.items():
        print(f"{ood_name}: TPR={metrics['TPR']:.2f}, FPR={metrics['FPR']:.2f}, AUC={metrics['AUC']:.2f}")


if __name__ == "__main__":
    # Run the experiment
    run_cifar10_experiment(data_dir="data", batch_size=64, alpha=0.05)
