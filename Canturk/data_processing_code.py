import pandas as pd

df = pd.read_json("hf://datasets/graphs-datasets/MUTAG/full.jsonl", lines=True)

df.to_csv("mutag_data.csv", index=False)

import ast

# Function to process edge_index and prevent symmetry
def process_edge_index_without_saving(edge_index_str):
    # Convert string representation of edge_index to a Python list
    edge_index = ast.literal_eval(edge_index_str)

    # Pair up source and target nodes
    edges = zip(edge_index[0], edge_index[1])

    # Deduplicate edges by using sorted tuples (to prevent symmetry)
    unique_edges = set((min(u, v), max(u, v)) for u, v in edges)

    # Convert back to edge_index format
    edge_index_unique = [[], []]
    for u, v in unique_edges:
        edge_index_unique[0].append(u)
        edge_index_unique[1].append(v)

    return edge_index_unique

df_edges=pd.read_csv("mutag_data.csv")

# Process the edge_index column to remove symmetry without saving to CSV
df_edges['processed_edge_index'] = df_edges['edge_index'].apply(process_edge_index_without_saving)

# Display the first few processed edge_index results
df_edges.head()

df_edges.to_csv("mutag_processed_edge_index.csv", index=False)

edges_list = []
final_list = []

# Iterate over the DataFrame column and sum the edges
for index, edges in df_edges.iterrows():
    processed_edges = edges['processed_edge_index']
    e_0 = processed_edges[0]
    e_1 = processed_edges[1]
    for i in range(len(e_0)):
        edges_list.append((e_0[i], e_1[i]))
    final_list.append(edges_list)
    edges_list = []

# Add the obtained edge list as a new column
df_edges['edges'] = final_list

print("Obtained edge list DataFrame:")
print(df_edges.head())

# Save the new DataFrame to a CSV file
df_edges.to_csv("mutag_edges_with_pairs.csv", index=False)

feature_list=df_edges["node_feat"].tolist()

import ast

# Convert the string to a list
feature_list_converted = [ast.literal_eval(item) for item in feature_list]

print(feature_list_converted)

print(feature_list_converted[0][0])
