import torch
import torch.nn as nn
import torch.nn.functional as F

i=0
k=0
adjacent_1=[]
adjacent_2=[]
adjacent_total=[]
features_1=[]
features_2=[]
features_total=[]

while i<188:
  while k<df_edges.iloc[i,4]:
    u_C_1=feature_list_converted[i][k]
    features_1.append(u_C_1)
    j=0
    while j<len(df_edges.iloc[i,6]):
      edge = df_edges.iloc[i, 6][j]

      if edge[0]==k:
        adjacent_1.append(edge[1])
        u_T_1=feature_list_converted[i][edge[1]]
        features_1.append(u_T_1)
      if edge[1]==k:
        adjacent_1.append(edge[0])
        u_T_1=feature_list_converted[i][edge[0]]
        features_1.append(u_T_1)
      j+=1
    adjacent_2.append(adjacent_1)
    adjacent_1=[]
    features_2.append(features_1)
    features_1=[]
    k+=1
  adjacent_total.append(adjacent_2)
  adjacent_2=[]
  features_total.append(features_2)
  features_2=[]
  k=0
  i+=1
print(adjacent_total)
print(k)
print(features_total[:3])

for i, item in enumerate(features_total):
    for j, sub_item in enumerate(item):
        print(f"features_total[{i}][{j}] boyutu: {len(sub_item)}")

# Make sub-elements fixed in size
fixed_padded_features_total = []

for item in padded_features_total:
    fixed_item = []
    for sub_item in item:
        # If sub_item is not a list, convert it to a 7-element list
        fixed_sub_item = [
            sub if isinstance(sub, list) else [0.0] * 7
            for sub in sub_item
        ]
        fixed_item.append(fixed_sub_item)
    fixed_padded_features_total.append(fixed_item)

features_total_tensor = torch.tensor(fixed_padded_features_total)

print("Tensor Shape:", features_total_tensor.shape)
print("Tensor:", features_total_tensor)

import torch
import torch.nn as nn
import torch.nn.functional as F

class HigherOrderInteraction(nn.Module):
    def __init__(self, d, r):
        super(HigherOrderInteraction, self).__init__()
        # Definition of parameters
        self.W2 = nn.Parameter(torch.randn(r, 2 * d))  # Weight matrix
        self.b = nn.Parameter(torch.randn(r))         # Bias vector
        self.a = nn.Parameter(torch.randn(r))         # Scalar weight vector

    def forward(self, u_C_1, u_T_1):
        """
        u_C_1: torch.Tensor, dimension (d,)
        u_T_1: torch.Tensor, dimension (d,)
        """
        # Concatenation of u(C_1) and u(T_1)
        concat = torch.cat((u_C_1, u_T_1), dim=-1)  # Boyut (2 * d,)

        # Transformation with Tanh activation function
        tanh_output = torch.tanh(torch.matmul(self.W2, concat) + self.b)  # Boyut (r,)

        # Scalar value computation
        scalar_value = torch.dot(self.a, tanh_output)  # Boyut (), skaler değer

        return scalar_value

d = 7
r = 3   # Hidden layer dimension

model = HigherOrderInteraction(d, r)

i=0
k=0
f_values_each_graph=[]
f_values_all=[]


while i<188:
  while k<df_edges.iloc[i,4]:
    u_C_1=features_total_tensor[i][k].mean(dim=0)
    j=0
    while j<len(df_edges.iloc[i,6]):
      edge = df_edges.iloc[i, 6][j]

      if edge[0]==k:
        adjacent_1.append(edge[1])
        u_T_1=features_total_tensor[i][edge[1]].mean(dim=0)
        f_value=model(u_C_1, u_T_1)
        f_values_each_graph.append(f_value)
      if edge[1]==k:
        adjacent_1.append(edge[0])
        u_T_1=features_total_tensor[i][edge[0]].mean(dim=0)
        f_value=model(u_C_1, u_T_1)
        f_values_each_graph.append(f_value)
      j+=1
    adjacent_total.append(adjacent_1)
    adjacent_1=[]
    f_values_all.append(f_values_each_graph)
    f_values_each_graph=[]
    k+=1
  adjacent_total.append(adjacent_2)
  adjacent_2=[]
  f_values_all.append(f_values_each_graph)
  f_values_each_graph=[]
  k=0
  i+=1
print(f_values_all)

all_alpha_values = []

for graph_f_values in f_values_all:  # The f_values_all just calculated
    graph_alpha_values = []
    for f_values in graph_f_values:
        f_values_tensor = torch.tensor(f_values)
        exp_f_values = torch.exp(f_values_tensor)
        alpha_values = exp_f_values / exp_f_values.sum()
        graph_alpha_values.append(alpha_values.tolist())
    all_alpha_values.append(graph_alpha_values)

print("Alpha Values for All Graphs:")
print(all_alpha_values)

def softmax_function_over_neighbors_and_center(f_values):
    """
    Calculates attention coefficients for neighbors and the central node.

    Args:
        f_values (list or tensor): The computed f(u(C_k), u(T_k)) values for the central node and neighbors.

    Returns:
        torch.Tensor: Attention coefficients normalized with Softmax.
    """
    f_values_tensor = torch.tensor(f_values, dtype=torch.float32)  # Convert to tensor
    exp_f_values = torch.exp(f_values_tensor)  # Apply exponential operation
    softmax_values = exp_f_values / exp_f_values.sum()  # Softmax normalization
    return softmax_values


for graph_index, (graph_alpha_values, graph_features) in enumerate(zip(all_alpha_values, features_total_tensor)):
    u_Ck_sa_graph = []  # To store the new values of each node

    for k, f_values in enumerate(f_values_each_graph):
        u_Ck = graph_features[k].mean(dim=0)  # Central node vector
        neighbors = df_edges.iloc[graph_index, 6]  # Edges in the current graph

        # Select only the neighbors connected to k from the neighbors list
        filtered_neighbors = [n[1] if n[0] == k else n[0] for n in neighbors if k in n]

        # Calculate attention coefficients (for neighbors and the center)
        alpha_values = softmax_function_over_neighbors_and_center(f_values)

        # Separate the attention coefficients of the neighbors
        neighbor_alphas = alpha_values[:-1]
        central_alpha = alpha_values[-1]

        # Compute the contribution of the neighbors
        neighbor_sum = torch.zeros(d_out)  # Toplama işlemi için başlangıç
        for neighbor_index, alpha in zip(filtered_neighbors, neighbor_alphas):
            u_Tk = graph_features[neighbor_index].mean(dim=0)
            neighbor_sum += alpha * W1 @ u_Tk

        # Contribution of the central node
        central_node_contribution = central_alpha * W1 @ u_Ck

        # Total new node vector
        u_Ck_sa = neighbor_sum + central_node_contribution
        u_Ck_sa_graph.append(u_Ck_sa)

    u_Ck_sa_all.append(u_Ck_sa_graph)

print(u_Ck_sa_graph)

def softmax_function_over_neighbors_and_center(f_values):
    f_values_tensor = torch.tensor(f_values, dtype=torch.float32)
    return F.softmax(f_values_tensor, dim=0)
