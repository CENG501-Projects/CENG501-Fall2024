# Higher-Order Interaction Goes Neural: A Substructure Assembling Graph Attention Network for Graph Classification

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

This paper is written by Jianliang Gao, Jun Gao, Xiaoting Ying, Mingming Lu and Jianxin Wang and published in IEEE Transactions on Knowledge and Data Engineering where volume, issue and date are 35, 2 and 1 February 2023, respectively. 

Data can come in very different forms such as point clouds or graphs. Numerous applications in cheminformatics or social networks use classification of graphs to get results. Hence, the existence of Neural Networks working on graph-structured data is essential and Graph Neural Networks (GNNs) covers this need. However, existing GNN models tries to capture the information of first-order neighboring nodes within a single layer and mostly don't give enough importance to graph substructure and substructure information. In this paper, the proposed method SA-GAT not only investigate and extract linear information from higher-order substructures but also focus on their non-linear interaction information via a core module "Substructure Interaction Attention" (SIA). 

My aim is to fully understand the method and obtain the same experimental results by providing open code for the sake of community.

## 1.1. Paper summary

The paper introduces a method contributing to Graph Neural Networks to classify graphs. It can be seen as a function from the input space of graphs to the set of graph labels. The aim is to learn this function making loss smallest along the network. The existing literature takes substructures into account, but this process usually just covers the immediate neighbors for each node within a single layer. Passing to higher-order substructures by increasing number of layers enables receptive field to enlarge, but it can lead to failure of convergence or decrease in performance. Later works including k-GNN and SPA-GAT enable considering higher-order substructures within a single layer. However, they fail to investigate mutual influence among them. This paper introduces SIA to eliminate this deficiency and uses Max and SSA Pooling to extract local and global feature embeddings of graphs.

# 2. The method and our interpretation

## 2.1. The original method

The method has four different parts. It starts with the input and higher-order layer. We are given a graph $G$ which is a pair $(V,E)$ with a set of nodes $V$ and a set of edges $E$ where $E \subseteq \{(i,j) | i,j \in V, i \noteq j\}.

## 2.2. Our interpretation

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
