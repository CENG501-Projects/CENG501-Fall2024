import matplotlib.pyplot as plt
from utils import calculate_sem
import numpy as np

def visualize_results(dataset_name, results):
    # Plot gains
    gains = {k: ((v[0] - results["erm"][0]) / results["erm"][0]) for k, v in results.items() if k != "erm" and v != []}
    plt.bar(gains.keys(), gains.values())
    plt.xlabel("LRW Method")
    plt.ylabel("Accuracy Gain Over ERM (%)")
    plt.title(f"Accuracy Gains on {dataset_name}")
    plt.savefig(f"{dataset_name}_gains.png")
    

def hard_margin_difference(base_margins, cifar_hard_margins):
    # Calculate statistics
    cifar_mean = np.mean(cifar_hard_margins - base_margins)
    cifar_median = np.median(cifar_hard_margins - base_margins)

    # Plot for CIFAR-100
    plt.figure(figsize=(6, 5))
    plt.hist(cifar_hard_margins - base_margins, bins=10, alpha=0.7, color='teal', edgecolor='black')
    plt.axvline(cifar_mean, color='blue', linestyle='-', label='mean')
    plt.axvline(cifar_median, color='red', linestyle='--', label='median')
    plt.title("CIFAR-100")
    plt.xlabel("margin difference b/w LRW_hard and ERM")
    plt.ylabel("Relative Counts")
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.savefig("cifar100_margin_diff.png")

def bucket_difference(base_margins, cifar_hard_margins, cifar_easy_margins):
    # Function to scale values to [-1, 1]
    def rescale_to_range(data, new_min=-1, new_max=1):
        old_min = np.min(data)
        old_max = np.max(data)
        return new_min + (data - old_min) * (new_max - new_min) / (old_max - old_min)

    # Rescale base_margins to [-1, 1]
    rescaled_base_margins = rescale_to_range(base_margins, -1, 1)
    rescaled_hard_margins = rescale_to_range(cifar_hard_margins, -0.2, 0.2)
    rescaled_easy_margins = rescale_to_range(cifar_easy_margins, -0.2, 0.2)
    # Define 10 buckets between -1 and 1
    bucket_edges = np.linspace(-1, 1, 11)  # 10 intervals mean 11 edges
    bucket_labels = (bucket_edges[:-1] + bucket_edges[1:]) / 2  # Midpoints of the buckets

    # Assign each base margin to a bucket
    bucket_indices = np.digitize(base_margins, bucket_edges, right=True)

    # Initialize lists to store means and SEMs
    hard_means = []
    hard_sems = []
    easy_means = []
    easy_sems = []
    diff_hard = cifar_hard_margins - base_margins
    diff_easy = cifar_easy_margins - base_margins
    # SEM calculation function
    def calculate_sem(data):
        if len(data) == 0:
            return 0
        std_dev = np.std(data, ddof=1)
        return std_dev / np.sqrt(len(data))

    # Calculate means and SEMs for each bucket
    for i in range(1, len(bucket_edges)):
        # Select data in the current bucket
        bucket_data_hard = diff_hard[bucket_indices == i]
        bucket_data_easy = diff_easy[bucket_indices == i]

        # Calculate statistics
        hard_means.append(np.mean(bucket_data_hard) if len(bucket_data_hard) > 0 else 0)
        hard_sems.append(calculate_sem(bucket_data_hard) if len(bucket_data_hard) > 0 else 0)

        easy_means.append(np.mean(bucket_data_easy) if len(bucket_data_easy) > 0 else 0)
        easy_sems.append(calculate_sem(bucket_data_easy) if len(bucket_data_easy) > 0 else 0)

    # Plot the results
    plt.figure(figsize=(8, 6))

    # Plot LRW-Hard
    plt.errorbar(bucket_labels, hard_means, yerr=hard_sems, fmt='-o', color='red', label='LRW-Hard')

    # Plot LRW-Easy
    plt.errorbar(bucket_labels, easy_means, yerr=easy_sems, fmt='-s', color='green', label='LRW-Easy')

    # Add labels, title, and legend
    plt.title("CIFAR-100")
    plt.xlabel("ERM Margin Buckets")
    plt.ylabel("Delta w.r.t. ERM")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cifar100_bucket_diff.png")