# Higher-Order Interaction Goes Neural: A Substructure Assembling Graph Attention Network for Graph Classification

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper "Higher-Order Interaction Goes Neural: A Substructure Assembling Graph Attention Network for Graph Classification" [1] was written by Jianliang Gao, Jun Gao, Xiaoting Ying, Mingming Lu and Jianxin Wang and published in IEEE Transactions on Knowledge and Data Engineering where volume, issue and date are 35, 2 and 1 February 2023, respectively. 

Data can come in very different forms such as point clouds or graphs. Numerous applications in cheminformatics or social networks use classification of graphs to get results. Hence, the existence of Neural Networks working on graph-structured data is essential and Graph Neural Networks (GNNs) covers this need. However, existing GNN models tries to capture the information of first-order neighboring nodes within a single layer and mostly don't give enough importance to graph substructure and substructure information. In this paper, the proposed method SA-GAT not only investigate and extract linear information from higher-order substructures but also focus on their non-linear interaction information via a core module "Substructure Interaction Attention" (SIA). 

My aim is to fully understand the method and obtain the same experimental results by providing open code for the sake of community.

## 1.1. Paper summary

The paper introduces a method contributing to Graph Neural Networks to classify graphs. It can be seen as a function from the input space of graphs to the set of graph labels. The aim is to learn this function making loss smallest along the network. The existing literature takes substructures into account, but this process usually just covers the immediate neighbors for each node within a single layer. Passing to higher-order substructures by increasing number of layers enables receptive field to enlarge, but it can lead to failure of convergence or decrease in performance. Later works including k-GNN and SPA-GAT enable considering higher-order substructures within a single layer. However, they fail to investigate mutual influence among them. This paper introduces SIA to eliminate this deficiency and uses Max and SSA Pooling to extract local and global feature embeddings of graphs.

# 2. The method and our interpretation

## 2.1. The original method

![Image](./images/Method.PNG)

The method has four different parts. It starts with the input and higher-order layer. We are given a graph $G$ which is a pair $(V,E)$ with a set of nodes $V$ and a set of edges $E$ where $E \subseteq \\{ (i,j) | i,j \in V, i \neq j \\} $. $V(G)$ and $E(G)$ denote the set of nodes and edges, respectively. Moreover, define the neighborhood of a node $i$ as $N(i) = \\{ j | (i,j) \in E(G) \\}$ and show the feature embedding encoding attributes of $i$ as $u_i$. After starting with such a graph $G$, substructures of $G$ which are $1$-order substructures are considered and they give $1$-order graph which is $G$ itself. To extract more information about $G$, higher-order graphs are defined in the following way.

**Definition.** For an integer $k \geq 2$, we denote any $k$ different connected nodes forming a connected subgraph in $G$ as $C_k=\\{v_1,\ldots,v_k\\}$. We identify $C_k$ as a node in $k$-order graph. $V(G)^k$ is denoted as the set of all nodes of $k$-order graph. The neighborhood of the node $C_k$ is defined as:
$$N(C_k)=\\{T_k \in V(G)^k | |C_k \bigcap T_k| = k-1\\}.$$

With this definition, we can create higher-order graphs for $k \geq 2$. 

### 2.1.1. Input and Higher-order Layer

The next step is to initialize node features of 1-order and higher order graphs. For $i \in V(G)$, the feature embedding $u_i \in ℝ^d$ is the concatanation of two one-hot vectors $e_i \in ℝ^{d_1}$ and $a_i \in ℝ^{d_2}$ based on label and attributes of the node $i$, respectively. Note that $d=d_1+d_2$. For node $C_k$ where $k \geq 2$,

```math
u(C_k)=\frac{1}{k}\sum_{C_1 \in C_k} u(C_1),
```
that is, $u(C_k)$ is just the average of feature embeddings of the nodes that constitute it.

### 2.1.2. Substructure Interaction Attention Layer

Second step of the method includes Substructure Interaction Attention (SIA) Layer. The main aim is to train $u(C_k)$ for each substructure $C_k \in V(G)^k$ and it is done with the contribution of two parts: the neighbor structure aggregation (sa) and the neighbor interaction aggregation (ia). Let

```math 
u(C_k)^{'}_{sa} \textrm{ and } u(C_k)^{'}_{ia}
```

denote the new representation of $u(C_k)$ obtained from neighbor aggregation and neighbor interaction aggregation, respectively.

### Neighboring Substructure Aggregation

To create it, we need to take all feature embeddings of neighbors of the node $C_k$ into account using attention mechanism:

```math 
u(C_k)^{'}_{sa} = \sum_{T_k \in N(C_k)} \alpha(C_k,T_k) W_1 u(T_k) + \alpha(C_k,C_k) W_1 u(C_k),
```

where $W_1 \in ℝ^{d' x d}$ is a shared weight matrix with the desired dimension $d'$ of $u(C_k)^{'}_{sa}$ and $\alpha(C_k,T_k)$ is the attention coefficient which is computed as follows:

$$
\alpha(C_k,T_k) = \frac{exp(f(u(C_k),u(T_k)))}{\sum_{T_k \in N(C_k) \cup \\{C_k\\}} exp(f(u(C_k),u(T_k)))},
$$

where $f$ is the following feed-forward network with a single hidden layer:

```math 
f(u(C_k),u(T_k))= a^{T} tanh(W_2U+b),
```

where $a \in ℝ^d$ is the weight vector to obtain a scalar as a result of applying $f$ and $a^T$ means the transpose of $a$. $W_2 \in ℝ^{r x 2d}$ is the weight matrix, $U \in ℝ^{2d x 1}$ is the matrix obtained by first putting each entry of $u(C_k)$ in rows by obeying the order of it, and then doing same process for $u(T_k)$. Moreover, $b \in ℝ^d$ is the bias and $r$ is the dimension size of the hidden layer.

### Neighboring Substructure Interaction Aggregation

The neighbor interaction representation of node $C_k$ is given as:

```math 
u(C_k)^{'}_{ia} = \sum_{T_k \in N(C_k)} \sum_{S_k \in N(C_k)} \beta(T_k,S_k)(u(T_k)*u(S_k)),
```

where $*$ is the element-wise multiplication operator and $\beta(T_k,S_k)$ denotes the interaction coefficient between nodes $T_k$ and $S_k$. If $(T_k,S_k) \in E(G)^k$, we define $\beta(T_k,S_k)=\alpha(C_k,T_k)\alpha(C_k,S_k)$. Otherwise, it equals to 0.

Instead, their normalized versions $\beta(T_k,S_k)^*$ can be used to make coefficients easily comparable:

```math 
\beta(T_k,S_k)^* = \frac{\beta(T_k,S_k)}{\sum_{M_k,Q_k \in N(C_k),(M_k,Q_k) \in E(G)^k} \beta(M_k,Q_k)}.
```

### Combining Two Parts

The new representation of $u(C_k)$ which is denoted as $u(C_k)'$ combining neighbor information and neighbor interaction information is defined as:

```math 
u(C_k)' = \sigma\big(\alpha u(C_k)^{'}_{sa} + \big(1-\alpha\big)u(C_k)^{'}_{ia}\big), 
```

where $\alpha$ is a parameter to balance information coming from two parts.

### 2.1.3. The Pooling Layer

To classify graphs, we need information based on graph embeddings. This paper fulfills this need by utilizing two graph-level embedding methods: Stability-based Substructure Attention (SSA) pooling and the Max pooling to characterize global and local graph features, respectively.

### SSA Pooling

Suppose that the number of substructures in the graph $G$ is $N$. Note that $G$ doesn't imply the started graph, that is, it can also be a $k$-order graph in this subsection. Let $V(G)$ denote the set of substructures in graph $G$ and $u_n \in ℝ^{1xd}$ show the substructure embedding of $n \in V(G)$. Then, the graph-level embedding $h_G$ is given as:

```math 
h_G = \sum_{n=1}^N \sigma\big(u_n ReLU\big(W_3\big(u_n^T-\frac{1}{N}\sum_{m=1}^N u_m^T\big)\big)\big)u_n,
```

where $W_3 \in ℝ^{dxd}$ is the learnable weight matrix.

### Max Pooling

Let $V_G \in ℝ^{Nxd}$ mean the matrix representation of substructure embeddings of $G$. We define $u_n=<m_{n1},m_{n2},\ldots,m_{nd}>$. The maximum pooling works on the matrix $V_G$ and takes the maximum of each column and concatenates all of them:

$$
u_G = \\{max_n (m_{n1}) || max_n (m_{n2}) || \dots || max_n (m_{nd})\\}
$$

for $n \in \\{1,2, \ldots, N\\}$.

### 2.1.4. The Multilayer Perceptron and Output Layer

We combine grpah-embeddings learned for different higher_order graphs. For a graph $G$, two graph-level embeddings $h_G$ and $u_G$ of it are obtained in the pooling layer. We concatenate them to obtain $f_G = h_G ||u_G$. Similarly, we can construct ${f_G}^(k)$ for $k$-order graph $G^(k)$ and concatenate all of them to obtain the final graph embedding $F_G$:

$$
F_G=\\{{f_G}^{(1)} || {f_G}^{(2)} || \ldots || {f_G}^{(k)}\\}.
$$

As a last step, we put $F_G$ into the MLP layer to predict the class of the graph: $\widehat y=MLP(F_G)$.

## 2.2. Our interpretation

### In the Figure

![Image](./images/Graph.PNG)

How to obtain higher-order substructures and $2$-order graph is obvious. However, the arrows and painted parts of $2$-order graph interaction needs need more explanation. When we try to get neighboring substructure interaction information for the substructure $CD$, we compute $\beta(T_2,S_2)$ where $T_2$ and $S_2$ are neighbors of $CD$. We know that $\beta(T_2,S_2)$ is non-zero only if $T_2$ and $S_2$ are connected with an edge. In our case, $AC$ and $BC$ are neighbors of $CD$ which are connected. Similar for the pair $DE$ and $DF$. Hence, these pairs are painted to emphasize their interaction on $CD$ with a red arrow showing their contribution. Moreover, black arrows indicate the neighbor relation. In this example, working on $CD$ is just a choice. The similar process will be executed for other substructures to get all the interaction information.

### About the Dimension of the Matrix $W_1$

In the neighboring substructure aggregation part of the SIA Layer, a shared weight matrix $W_1 \in ℝ^{d'xd}$ is used. After the computations, dimension of $u(C_k)^{'}_{sa}$ turns into $d'$. On the other hand, the dimension for interaction is $d$ and we add them up to update the substructure embedding of $C_k$. If $d'=d$, there is nothing to be worried about. Otherwise, a modification is needed to equalise them. One idea is that extending less dimensional information with necessary number of zeros until equalising dimensions.

### Obtaining Global and Local Features of the Graph

It is important to understand how to obtain $u(C_k)$ from the labels and attributes of $C_k$. In the method, it is stated that $u(C_k)$ is constructed from the concatenation of two one-hot vectors constructed from labels and attributes of it. Although obtaining one-hot vector for labels is obvious, for attributes the process is unclear. It depends on the type of the attributes. For instance, if all attributes are categorical, one-hot vector representation of them is valid, but in the case of a numerical attribute, attainability of $u(C_k)$ is controversial.

After learning $u(C_k)^{'}$, the crucial step is to obtain graph embeddings using substructure embeddings. The two methods SSA Pooling and Max Pooling enables us to learn more about graph. SSA Pooling activates the attention mechanism to create an attention to each embedding $u(C_k)$ and combining all the information coming from substructures with attentions leads to learn global features of the graph. Furthermore, Max Pooling selects the maximum values from the components and facilitates highlighting the most prominent local features.

### Processing Data

We first considered building the model over the dataset "MUTAG" which has $188$ graphs with total number of $3371$ nodes and $7442$ edges. Data is obtained as a data frame with column labels as follows:
-"edge_index": for each graph, it consists of a list of two lists where corresponding indexes of two lists are connected with an edge. However, the symmetry is not exluceded. We see the same edge two times in the list.
-"node_feat": for each node in a graph, there exists a one-hot vector standing as $1x7$ matrix keeping the type of the node such as a carbon or an oxygen atom.
-"edge_attr": it will not be used in the proposed model.
-"y": it shows the classified class of the graph as "0" or "1".
-"num_nodes": it keeps number of the nodes in each graph.

We needed to get rid of from the symmetry in the column "edge_index" not only to obtain an easier interpretation, but also to construct higher-order graphs. We achieved this and kept the information in new column with name "processed_edge_index". Moreover, in this paper, one of the key points is to pass higher-order graphs from the started one. Hence, we extracted the information of all edges as tuples and contained them in a new column of the data frame with name "edges". In the next step, we will create $2$-order and $3$-order graphs by using them.

# 3. Experiments and results

## 3.1. Experimental setup

We concentrated building the model first for $1$-order graphs which are directly given ones. Since node features are stored in the form of strings, we converted that column to a list to reach feature embedding of each node. In the model, we will update each feature embedding by taking neighbors' embeddings into account. Hence, we created a list called "adjacent_total" to keep which nodes are connected with which nodes. For instance, $0$-th index of $0$-th index of the list shows first node (with number $0$) of the first graph is connected to nodes with number $5$ and $1$. Then, a list called "features_total" is constructed to keep corresponding features of the nodes. Hence, we were ready to use neighbors' features to obtain Neighboring Substructure Aggregation.

We tried to write the necessary code for substructure aggregation part. Attention between a node and each of its neighbors is obtained with proper results. However, our model for $u(C_k)^{'}_{sa}$ couldn't come up with meaningful results yet. What we have done can be followed via the files: "data_processing_code.py" and "model.py".

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

No results yet.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

1. J. Gao, J. Gao, X. Ying, M. Lu and J. Wang, "Higher-Order Interaction Goes Neural: A Substructure Assembling Graph Attention Network for Graph Classification," in IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 2, pp. 1594-1608, 1 Feb. 2023, doi: 10.1109/TKDE.2021.3105544. keywords: {Kernel;Graph neural networks;Aggregates;Task analysis;Feature extraction;Proteins;Knowledge discovery;Graph classification;graph attention networks;higher-order graph;substructure interaction},

# Contact

Uğur Bektaş Cantürk - ugur98liv92@gmail.com
