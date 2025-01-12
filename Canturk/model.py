import random
import numpy as np
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

#####

# Padding operation to equalize row and column sizes for each feature
max_features = max(len(feature) for sample in features_total for feature in sample)  # Max feature length
max_rows = max(len(sample) for sample in features_total)  # Max row length

# Padding process
padded_features = []
for sample in features_total:
    padded_sample = []
    for feature in sample:
        # We pad each feature to match the max_features length
        padded_feature = feature + [[0.0] * 7] * (max_features - len(feature))
        padded_sample.append(padded_feature)
    # We pad the rows to match the max_rows length
    while len(padded_sample) < max_rows:
        padded_sample.append([[0.0] * 7] * max_features)
    padded_features.append(padded_sample)

# Converting to a Torch tensor
features_total_tensor = torch.tensor(padded_features)

print(features_total_tensor.shape)  # Checking tensor dimensions

print(features_total_tensor[0])

#####

# Function to fix randomness sources
def set_seed(seed=42):
    """Fix randomness sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Definition of the HigherOrderInteraction class
class HigherOrderInteraction(nn.Module):
    def __init__(self, d, r):
        super(HigherOrderInteraction, self).__init__()
        self.W2 = nn.Parameter(torch.randn(r, 2 * d))  # Weight matrix
        self.b = nn.Parameter(torch.randn(r))         # Bias vector
        self.a = nn.Parameter(torch.randn(r))         # Scalar weight vector

    def forward(self, u_C_1, u_T_1):
        # Combining u(C_1) and u(T_1)
        concat = torch.cat((u_C_1, u_T_1), dim=-1)  # Shape (2 * d,)
        
        # Transformation and scalar value computation
        tanh_output = torch.tanh(torch.matmul(self.W2, concat) + self.b)  # Shape (r,)
        scalar_value = torch.dot(self.a, tanh_output)  # Scalar value
        
        return scalar_value

# Set the seed
set_seed(42)

# Tensor dimensions
d = 7   # Vector size
r = 3   # Hidden layer size

# Model definition
model = HigherOrderInteraction(d, r)

# Example tensors
results_tensor = torch.zeros(128, 28, 5, 1)  # Initialized with zeros: [128, 28, 5, 1]

# Loop: Compute u(C_1) (self) and u(T_1)
for i in range(features_total_tensor.shape[0]):  # 0-127 (128 examples)
    for j in range(features_total_tensor.shape[1]):  # 0-27 (28 rows)
        u_C_1 = features_total_tensor[i, j, 0]  # First row (u_1: self)

        for k in range(5):  # 0, 1, 2, 3, 4
            u_T_1 = features_total_tensor[i, j, k]  # Select T_k
            
            # If u(T_k) is completely zero, the result remains 0
            if not torch.all(u_T_1 == 0):  # Only if u(T_k) is not zero
                result = model(u_C_1, u_T_1)  # Compute result using the model
                results_tensor[i, j, k, 0] = result  # Save the result

# Check the tensor
print("Results Tensor Shape:", results_tensor.shape)  # Expected: [128, 28, 5, 1]
print("Results Tensor Example:", results_tensor[0, 0])  # Results of the first row of the first example

#####

# Example input: results_tensor

# Tensor to store the softmax result
alpha_tensor = torch.zeros_like(results_tensor)  # Result tensor: [128, 28, 5, 1]

# Softmax operation (tensor-based)
softmax_scores = F.softmax(results_tensor.squeeze(-1), dim=2)  # Shape: [128, 28, 5]
alpha_tensor = softmax_scores.unsqueeze(-1)  # Reshape back to [128, 28, 5, 1]

# Check the tensor
print("Alpha Tensor Shape:", alpha_tensor.shape)  # Expected: [128, 28, 5, 1]
print("Alpha Tensor Example:", alpha_tensor[0, 0])  # Results of the first row of the first example

#####

set_seed(42)

# W1 weight matrix (linear transformation)
W1 = nn.Parameter(torch.randn(7, 7))  # Learnable weight: [7, 7]

# Result tensor: u(C_1)'_sa will be calculated
u_C1_prime_sa = torch.zeros(128, 28, 7)  # Output: [128, 28, 7]

# Calculation: u(C_1)'_sa
for i in range(features_total_tensor.shape[0]):  # 128 examples
    for j in range(features_total_tensor.shape[1]):  # 28 rows
        u_C1 = features_total_tensor[i, j, 0]  # u(C_1) (itself)

        sum_weighted = torch.zeros(7)  # Empty vector for summation

        for k in range(5):  # T_1 values (0, 1, 2, 3, 4)
            u_T1 = features_total_tensor[i, j, k]  # u(T_1)
            alpha = alpha_tensor[i, j, k, 0]  # alpha(C_1, T_1)
            sum_weighted += alpha * torch.matmul(W1, u_T1)  # alpha * W1 * u(T_1)

        # Transformation for itself: alpha(C_k, C_k) * W1 * u(C_1)
        alpha_self = alpha_tensor[i, j, 0, 0]  # alpha(C_1, C_1)
        sum_weighted += alpha_self * torch.matmul(W1, u_C1)

        # Save the result
        u_C1_prime_sa[i, j] = sum_weighted

# Check the tensor
print("u_C1_prime_sa Shape:", u_C1_prime_sa.shape)  # Expected: [128, 28, 7]
print("u_C1_prime_sa Example:", u_C1_prime_sa[0])  # Results of the first row of the first example

######## For each graph, only the first number of nodes will be selected; we will not use all 28 rows.
