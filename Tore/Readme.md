# Transformers learn through gradual rank increase

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper titled "Transformers Learn Through Gradual Rank Increase" by Enric Boix-Adserà, Etai Littwin, Emmanuel Abbe, Samy Bengio, and Joshua Susskind was published on arXiv (2306.07042v2) on December 12, 2023. It is part of Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Main Conference Track. It explores the training dynamics of transformer models, focusing on the phenomenon of incremental learning where the rank of the difference between trained and initial weights increases gradually during training. This dynamic is rigorously analyzed in a simplified theoretical setting and supported by experiments on practical transformer architectures, including Vision Transformers (ViTs) trained on datasets like CIFAR-10 and ImageNet.
The paper demonstrates that transformers exhibit low-rank bias in weight updates, even in practical settings, without explicitly enforcing it. These findings could improve our understanding of why transformers are so effective and help optimize methods like LoRA (Low-Rank Adaptation).

The goal of this work is to reproduce the key results of the paper, ensuring that the experimental findings are accurate and replicable. This involves:
1.	Simulating Simplified Incremental Learning Dynamics: Reproducing the theoretical setup with diagonal weight matrices and verifying the incremental rank increase under small initialization.
2.	Training Practical Transformer Models:Training Vision Transformers (ViTs) on datasets like CIFAR-10 and ImageNet. Monitoring the rank of weight perturbations (e.g., $W_K W_Q^⊤$) at different iterations during training.
3.	Visualizing Results:Generating spectra plots similar to those in the paper, showing the normalized spectra of weight perturbations across iterations.
By successfully replicating the results, we aim to verify the claims in the paper and provide a foundation for further exploration into training dynamics and their implications for model efficiency.

## 1.1. Paper summary

The paper investigates the incremental learning dynamics of transformer models, where the rank of the difference between trained and initial weights increases progressively during training. The authors present both theoretical insights and empirical evidence to support their findings, shedding light on the structured nature of transformer training dynamics .

The authors simplify the transformer model by assuming diagonal weight matrices for attention layers and small initialization.
They prove that training proceeds in stages, where weights plateau near a saddle point for most of a stage. At the end of each stage, the rank of weight updates increases by at most one. These findings are derived using gradient flow dynamics and extend existing theories from simpler, linear networks to nonlinear transformer models.

The model contributes the literature in theoretical and empirical result related to how transformers work. The paper identifies and rigorously analyzes the incremental rank growth during transformer training, previously unexplored in nonlinear attention-based models. It extends theories of incremental learning[[4]](#4) and low-rank bias from simpler linear networks to more complex nonlinear architectures, such as transformers. In empirical side, the paper demonstrates that incremental rank growth occurs in practice for models trained with standard optimizers (e.g., Adam), even when theoretical assumptions (e.g., diagonal weights) do not apply. They link their findings to LoRA [[1]](#1) , a fine-tuning method that constrains weight updates to low-rank subspaces, suggesting that the incremental dynamics observed in this work might explain LoRA's efficiency.
# 2. The method and our interpretation

## 2.1. The original method

The paper explores the incremental learning dynamics of transformers, describing how weight updates evolve in a structured, stage-wise manner during training. The method is a blend of theoretical analysis for simplified transformers and empirical validation on real-world models.

1. Simplified Theoretical Framework
The authors analyze transformer training dynamics under two simplifying assumptions:

Diagonal Weights: Each attention head’s weight matrices (
$𝑊_𝐾 , 𝑊_𝑄 , 𝑊_𝑉 , 𝑊_𝑂$ ) are diagonal.
Small Initialization: Weights are initialized with very small values (∼𝑂(𝛼), where 𝛼≪1).  [[3]](#3)
Using these assumptions, they derive:

Discrete Stages in Training:

Training progresses in stages. During each stage:
Weights plateau near a saddle point for most of the time.
At the end of the stage, the rank of the weight update 
$Δ𝑊=𝑊_{trained}−𝑊_{initial}$ increases by at most one.
Key Results:

For the simplified diagonal transformer:
$𝑊_𝐾 𝑊_𝑄^⊤$  and $𝑊_𝑉 𝑊_𝑂^⊤$ incrementally increase in rank by one at the end of each stage.
These dynamics generalize to nonlinear networks, extending prior works focused on linear models.

Mathematical Formulation:

The method uses gradient flow [[2]](#2) to track the evolution of weights:

$𝑑𝜃𝑑𝑡=−∇_𝜃 𝐿(𝜃)$
Analysis reveals that weight updates are biased toward low-rank solutions, and rank increases step-by-step as the model escapes saddle points.

2. Empirical Validation in Real Transformers

The authors apply their theoretical insights to practical transformers, such as Vision Transformers (ViTs), and measure the dynamics of weight perturbations during training.

Training Real Transformers:

Models are trained on datasets like CIFAR-10 and ImageNet using standard optimizers (e.g., Adam).
The weight perturbations $Δ𝑊=𝑊_{trained}−𝑊_{initial}$​ are tracked for attention layers.

Analyzing Rank Growth:

The rank of the perturbation matrix $𝑊_𝐾 𝑊_𝑄^⊤$ is computed at multiple iterations using singular value decomposition (SVD):

Rank(𝑊)=number of non-zero singular values of 𝑊

Results show a gradual increase in rank during training, consistent with theoretical predictions.
Normalization and Spectra Analysis:

Spectra of weight perturbations are plotted, showing the normalized singular values at initialization and at various training stages.

## 2.2. Our interpretation

Although the paper defines the experimental setting in detail, some of the things need to be clarified.

The paper does not specify whether exact rank or stable rank is used. Both are computed for robustness.
Imagenet set used is not specified in the paper, due to dataset size consideration Hugginface Imagenet 1000k set will be used in our analysis.

1. Definition of Incremental Stages

Original Paper:
The paper describes incremental learning stages where the rank of the weight update matrix increases by at most one during each stage. However:
It does not clearly define how to identify the start and end of a stage in empirical settings.
The theoretical framework relies on gradient flow (an idealized version of optimization), but it is unclear how to relate this to stochastic gradient descent (SGD) or Adam in practice.
Interpretation:
Stage Detection:

Empirically, a "stage" is inferred by tracking the rank of the perturbation matrix  over iterations. A sharp increase in rank suggests the end of one stage and the start of another.
Loss plateaus, as described in the theory, may also indicate stages, but these are not explicitly linked to rank changes in the experiments.
Gradient Flow Approximation:
Treat SGD/Adam as approximating gradient flow but introduce noise due to stochastic updates. Rank growth patterns are expected to remain qualitatively similar despite these differences.

2. Experimental Design Details

Original Paper:
The experiments lack detailed descriptions of:
Training settings (e.g., batch size, learning rate, optimizer parameters).
Specific iterations or checkpoints used for analyzing rank changes.
Interpretation:
Training Settings:
Use standard practices for Vision Transformers:
Optimizer: Adam with learning rate 10^−4$.
Batch size: 32.
Number of iterations: Up to 50,000 (track at key points like 1, 10, 50, etc.).
Adjust hyperparameters to align with the size of CIFAR-10 or ImageNet datasets.
Iteration Tracking:
Track weight updates and perturbations at predefined iterations, as these are likely points used in the paper (e.g., 1, 10, 50, 100, etc.).

# 3. Experiments and results

## 3.1. Experimental setup

The paper investigates incremental rank growth in transformer models using two complementary approaches: a simplified theoretical setup and experiments on practical transformer models.

1. Simplified Theoretical Setup
Goal: Prove incremental learning dynamics under idealized conditions.
Model Assumptions:
Diagonal Weights: Attention head weights $(𝑊_𝐾,𝑊_𝑄,𝑊_𝑉,𝑊_𝑂)$ are diagonal matrices.
Small Initialization: Weights are initialized close to zero (∼O(α), where 𝛼≪1.
Training Assumptions:
Training is modeled using gradient flow dynamics:
$𝑑𝜃𝑑𝑡=−∇_𝜃 𝐿(𝜃)$
Loss plateaus during most of each stage, and the rank of weight updates increases by at most one at the end of each stage.

3. Practical Experimental Setup
Models:

Vision Transformer (ViT) trained on CIFAR-10, CIFAR-100, and ImageNet.
GPT-2 trained on Wikitext-103.
Training Configuration:

Optimizer: Adam with standard parameters.
Loss Function: Cross-Entropy Loss.
Datasets:
CIFAR-10/CIFAR-100 resized to 
224
×
224
224×224 for ViT input.
ImageNet for ViT.
Wikitext-103 for GPT-2.
Layer Selection: Focused on specific attention heads (e.g., 
$𝑊_𝐾 𝑊_𝑄^⊤$).
Rank Analysis:

The rank of the weight perturbation matrix 
$Δ 𝑊 =𝑊_{trained} −𝑊_{initial}$ is tracked over iterations.
Singular Value Decomposition (SVD) is used to compute spectra and stable rank.
Visualization:

Normalized spectra of weight perturbations are plotted at initialization and during training (e.g., Figure 1).
Changes to the Setup
While aiming to replicate the original results, some adjustments were made to fill in missing details or adapt the setup:

1. Theoretical Setup Adjustments
Gradient Flow:
Direct implementation of gradient flow (𝑑𝜃𝑑𝑡) is infeasible in standard deep learning libraries, so stochastic gradient descent (SGD) or Adam is used as an approximation.

Diagonal Weights:
Practical transformers do not use diagonal weights. Instead, their full matrices are analyzed.

Initialization:
To mimic small initialization, initial weights are scaled by a factor ($α∼10^{−2}$) before training.
2. Practical Setup Adjustments
Datasets:

CIFAR-10 is used primarily due to computational constraints. CIFAR-100 and ImageNet are omitted in some experiments.
Image resolution is adjusted to 224×224 for ViT input.

Model:

A smaller ViT configuration (e.g., fewer attention heads or layers) is used to reduce computational requirements.

Iterations for Analysis:

Perturbation analysis is conducted at predefined checkpoints (e.g., 1, 10, 50, 100 iterations) instead of continuous tracking.
Layer Focus:

Attention heads in the first layer (‘vit.encoder.layer.0.attention‘) are analyzed as a representative case, instead of tracking all layers.
3. Analysis Adjustments

Rank Calculation:
The paper does not specify whether exact rank or stable rank is used. Both are computed for robustness:
Exact Rank: Based on the count of non-zero singular values.
Stable Rank: Stable Rank $=∥𝑊∥_𝐹^2/∥𝑊∥_2^2$.

Normalization:

Singular values are normalized relative to the largest singular value to align with the visualizations in the paper.


## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

![image](https://github.com/user-attachments/assets/91232a6c-cf38-44b8-a734-0db85a94c28f)
Figure 1: For an attention head in ViT trained on (a) CIFAR-10, we plot the normalized spectra of $W_KW^⊤_Q$ at initialization (in red), and of the learned perturbations to WKW⊤ Q at different iterations (in green).

![image](https://github.com/user-attachments/assets/ca8c7793-d5fd-4032-a18c-256d7a1abfd6)
![image](https://github.com/user-attachments/assets/e107d933-225e-4e58-951a-abb5f16751fb)
![image](https://github.com/user-attachments/assets/332903ee-cb73-444f-a2c0-4ff5c7437739)
Figure 2: (a) Loss versus rescaled time in the toy task of learning an attention head with diagonal weights, for various initialization scales α. The loss curves converge as α → 0 to a curve with stagewise loss plateaus and sharp decreases, as predicted by the theory; some stagewise learning behavior is already clear with α = 0.01. (b) Each line shows the evolution of one of the entries of diag(wQ)diag(wK) and diag(wV )diag(wO) over rescaled time, demonst

![image](https://github.com/user-attachments/assets/b43e744a-65c4-42a9-9d54-3968f2a9f5b5)
Figure 3: Validation of assumptions on the toy model of learning a single attention head. (a) Assumption 4.4: weights perturbed at a random time during training (solid lines) tend back to the near-stationary point (dashed lines). (b) Assumption 4.5: weights perturbed at the beginning of a stage (solid lines) have same nonlinear evolution a
ViT, ImageNet (d) GPT-2, Wikitext-103

![image](https://github.com/user-attachments/assets/6711ba82-448b-494f-8530-89eee3c7c702)
Figure 4: Stable rank of $∆W_KW^⊤_Q$ (blue) and $∆W_V W^⊤_O$
(orange) on an arbitrary chosen layer
throughout training for four different pairs of networks and tasks. The stable rank of a matrix W
is defined as $∥W∥^2_F/∥W∥^2_2$, and gives a smooth approximation of the rank. Mean and standard
deviation (shaded area) are computed ac
# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

#### <a id="1">[1]</a> Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, Lora: Low-rank adaptation of large language models, arXiv preprint arXiv:2106.09685 (2021).
#### <a id="2">[2]</a> Francis Bach, Effortless optimization through gradient flows, Machine Learning Research Blog. https://francisbach. com/gradient-flows (2020). 
#### <a id="3">[3]</a> Arthur Jacot, Francois Gaston Ged, Berfin Simsek, Cl´ement Hongler, and Franck Gabriel, Saddle-to-saddle dynamics in deep linear networks: Small initialization training, symmetry, and sparsity, 2021.
#### <a id="4">[4]</a> Jiawei Zhao, Yifei Zhang, Beidi Chen, Florian Sch¨afer, and Anima Anandkumar, Inrank: Incremental low-rank learning, arXiv preprint arXiv:2306.11250 (2023).

# Contact

Nurşen Töre email:nursen.tore@metu.edu.tr
