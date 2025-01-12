# take 1 exp output -> plot graph

# graph types:
# 1 - normal train graph
# 2 - sensitivity vs to %90 accuracy
# 3 - sensitivity vs current accuracy 
# 4 - complexity vs %90 accuracy

import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import io

class Config:
    def __init__(self, input_dict) -> None:
        for top_key, sub_dict in input_dict.items():
            if isinstance(sub_dict, dict):
                for sub_key, value in sub_dict.items():
                    if value == "nil":
                        value = None
                    setattr(self, sub_key, value)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_process_pkl(path, file):
    file_path = os.path.join(path,file)
    with open(file_path, "rb") as f:
        run_data = CPU_Unpickler(f).load()
    return run_data

def graph_train(x, y):
    pass

def graph_complexity_vs_acc(x, y):
    pass

def graph_synonym_vs_acc(x, y):
    pass

def graph_diffeo_vs_acc(x, y):
    pass


def get_sample_complexity(x_list, y_list, target_acc):
    if target_acc < 1:
        raise ValueError("Accuracy values are between 0-100, you probably forgot that")
    for x, y in zip(x_list,y_list):
        if y >= target_acc:
            return x
    return None


def plot_log_log_performance(data):
    """
    Creates a log-log scatter plot with a dashed x = y line.

    Parameters:
    data (list of tuples): List of (x, y, v, s0, L) tuples, where:
        - x: Number of data points seen when 90% accuracy is achieved.
        - y: Number of data points seen when 10% sensitivity is achieved.
        - v: Feature multiplicity (determines opacity).
        - s0: Feature sparsity (determines marker style).
        - L: Number of layers (determines color: 2 = red, 3 = blue).
    """
    # Unpack the data
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    v = [d[2] for d in data]
    s0 = [d[3] for d in data]
    L = [d[4] for d in data]

    # Create a DataFrame for seaborn
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'v': v,
        's0': s0,
        'L': L
    })

    # Map L to colors
    # df['color'] = df['L'].map({2: 'red', 3: 'blue'})

    # Map s0 to markers
    markers = {0: '^', 1: 's', 2: 'o', 4: 'D'}
    df['marker'] = df['s0'].map(markers)

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Plot each point with fthe appropriate color, marker, and opacity
    for marker_type in markers.values():
        subset = df[df['marker'] == marker_type]
        plt.scatter(subset['x'], subset['y'], alpha=0.1 + (1/30) * subset['v'], 
                    marker=marker_type, s=100, label=f's0={subset["s0"].iloc[0]}')

    # Add the x = y line
    max_val = max(max(x), max(y))
    plt.plot([1, max_val], [1, max_val], 'k--', label='x = y')

    # Set log-log scale
    plt.xscale('log')
    plt.yscale('log')

    # Add labels and title
    plt.xlabel('Theoretical Complexity - log scale')
    plt.ylabel('Sample Complexity To (90% accuracy) - log scale')
    plt.title('Performance Plot with Feature Multiplicity and Sparsity (Log-Log Scale)')

    plt.xlim(10, max(x)*2)
    plt.ylim(10, max(y)*2)

    # Add legend
    plt.legend(title="Feature Sparsity (s0)")

    # Show the plot
    plt.grid(True, which="both", ls="--")
    plt.show()

def load_for_sensitivity_plot(path, prefix):
    files = [file for file in os.listdir(path) if file.startswith(prefix)]
    loads = []
    for file in files:
        obj = load_process_pkl(path, file)
        loads.append(obj)
    return loads

def get_theoretical_complexity(obj):
    s = obj["args"].s
    L = obj["args"].num_layers
    s0 = obj["args"].s0
    nc = obj["args"].num_classes
    m = obj["args"].m
    if obj["args"].net == "lcn":
        return (s**(L/2)) * ((s0+1)**L) * nc * (m**L)
    elif obj["args"].net == "cnn":
        return ((s0+1)**2) * nc * (m**L)
    
    return None

def main():
    data = load_process_pkl("outputs", "exp3a_config_2_12.pkl")
    print(vars(data["args"]))
    

    train_size = vars(data["args"])["ptr"]
    test_size = vars(data["args"])["pte"]

    x_list = []
    y_list = []
    print(data)
    for point in data["dynamics"]:
        if "sensitivity_diffeo" in point:
            print("Here")
            x, y = point["sensitivity_diffeo"], point['acc']
            x_list.append(x)
            y_list.append(y)
        if "sensitivity_synonym" in point:
            print("Here")
            x, y = point["sensitivity_diffeo"], point['acc']
            x_list.append(x)
            y_list.append(y)
    print(f"Train Size : {train_size}")
    print(f"Test Size : {test_size}")

    total_epochs = data["epoch"][-1]
    total_p = total_epochs*train_size
    print(f"Total data seen : {total_p}")

    # Plotting the data
    df = pd.DataFrame({'Training Examples': x_list, 'Error Rate': [100-y for y in y_list]})    
    sns.lineplot(x='Training Examples', y='Error Rate', data=df)  # ci='sd' shows the standard deviation

    plt.xscale('log')
    plt.xlim(100, max(x_list))
    plt.axhline(y=10, color='red', linestyle='--', label='10% Error')
    # Adding labels and title
    plt.xlabel('Number of Training Examples Seen')
    plt.ylabel('Error Rate (%)')
    plt.title('Training Examples vs Error Rate')

    sample_complexity = get_sample_complexity(x_list, y_list, 90)
    print(f"Sample complexity is {sample_complexity}")

    # Display the plot
    plt.show()

    # objects = load_for_sensitivity_plot("outputs","exp3b_config")
    # data_list = []
    # for idx, obj in enumerate(objects):
    #     train_size = obj["args"].ptr
    #     v = obj["args"].num_features
    #     num_layers = obj["args"].num_layers
    #     s0 = obj["args"].s0
    #     if v == 10 and num_layers == 2 and s0 == 2:
    #         print(f"At {idx}")
    #     x_list = []
    #     y_list = []
    #     sensitivity_list = []
    #     for point in obj["dynamics"]:
    #         x, y = point['epoch']*train_size, point['acc']
    #         x_list.append(x)
    #         y_list.append(y)
    #         if "sensitivity_diffeo" in point and point["sensitivity_diffeo"]:
    #             sensitivity_list.append(point["sensitivity_diffeo"])
    #         else:
    #             sensitivity_list.append(0)

    #     sample_complexity = get_sample_complexity(sensitivity_list, y_list, 90)
    #     if not sample_complexity:
    #         continue
    #     theoretical_complexity = get_theoretical_complexity(obj)
    #     # print("")
    #     # print(f"net : {obj['args'].net}")
    #     # print(f"Dataset size : {obj['args'].ptr}")
    #     # print(f"Actual size : {train_size}")
    #     # print(f"theoretical_complexity : {theoretical_complexity}")
    #     # print(f"sample_complexity : {sample_complexity}")
    #     # print(f"v : {v}")
    #     # print(f"s0 : {s0}")
    #     # print(f"num_layers : {num_layers}")
    #     data_point = (theoretical_complexity, sample_complexity, v, s0, num_layers)
    #     data_list.append(data_point)
    # plot_log_log_performance(data_list)




if __name__ == "__main__":

    main()