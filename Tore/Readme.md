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

For running the code in Results.jpynb you need to download the dataset in the first section. For figures, you can run them seperately in each section dedicated to figures.

## 3.3. Results

Figure 1 illustrates the evolution of the eigenvalue spectra of attention weights in ViTs during training, revealing the emergence of a low-rank bias as the model learns to focus on dominant patterns in the data.
The figure highlights differences in training dynamics between CIFAR-10 (simpler dataset). In our study we are able to duplicate similar pattern for CIFAR10 data. Replicating for Imagenet Data requires time, so we are not able to do the run in specified time.
The spectrum at initialization (red) has a nearly flat distribution, suggesting that the initial 
$W_KW^⊤_Q$ is not biased towards any specific patterns or directions in the feature space.
As training progresses, the green curves (perturbation spectra) exhibit a low-rank bias:
The top eigenvalues grow significantly, while lower eigenvalues decay or remain small.
This indicates that the model is learning to focus on a few key directions in the attention mechanism, effectively reducing the rank of 
$W_KW^⊤_Q$

![image](https://github.com/user-attachments/assets/91232a6c-cf38-44b8-a734-0db85a94c28f)

Figure 1: For an attention head in ViT trained on (a) CIFAR-10, we plot the normalized spectra of $W_KW^⊤_Q$ at initialization (in red), and of the learned perturbations to WKW⊤ Q at different iterations (in green).

Figure 2 illustrates the training dynamics in a toy task designed to study the behavior of attention heads with diagonal weights. The figure contains two main components:
(a) Training Loss vs. Rescaled Time for Various Initialization Scales (𝛼)
Plot Description:
The plot shows the training loss as a function of rescaled time (time/log(1/α)) for different initialization scales (𝛼).
Stagewise Behavior: For smaller values of α, the training loss decreases in stagewise steps, with clear plateaus and sharp decreases between stages.
Convergence Across Scales: As α→0, all curves converge to a universal trajectory in rescaled time. This suggests that the training dynamics are scale-invariant when plotted against rescaled time.
Effect of α: Larger initialization scales (e.g., α=0.1) show smoother training dynamics, while smaller scales exhibit pronounced stagewise behavior
This behavior is consistent with theoretical predictions for low-rank learning, where training dynamics are influenced by the scale of initialization. Smaller initialization scales lead to more pronounced stagewise learning. We are able to replicate the paper results.

![image](https://github.com/user-attachments/assets/14d07de2-aa95-45ff-9d89-7cf004201116)
![image](https://github.com/user-attachments/assets/92e57351-e812-4cba-b4a1-454b48bcd7d4)

Figure 2: (a) Loss versus rescaled time in the toy task of learning an attention head with diagonal weights, for various initialization scales α. The loss curves converge as α → 0 to a curve with stagewise loss plateaus and sharp decreases, as predicted by the theory; some stagewise learning behavior is already clear with α = 0.01. (b) Each line shows the evolution of one of the entries of diag(wQ)diag(wK) and diag(wV )diag(wO) over rescaled time, demonst

Figure 3 demonstrates the validation of theoretical assumptions (Assumptions 4.4 and 4.5) on the toy model of learning a single attention head, focusing on the evolution of weights in the matrices 
$𝑊_𝑉𝑊_𝑂^⊤$.
Solid lines represent the evolution of individual entries in 
𝑊𝑉𝑊𝑂⊤  after a random perturbation is applied during training.
Dashed lines represent the corresponding evolution if no perturbation was applied.
The x-axis represents rescaled time (time/log(1/α)), and the y-axis represents the values of specific coordinates (diagonal entries) in 
$𝑊_𝑉𝑊_𝑂^⊤$.
Training is stable against random perturbations during and at the beginning of a stage, with weights returning to or following expected trajectories.
This indicates that the learning process is robust and consistent, even when disturbed. We are able view this behaviour in our study.

![image](https://github.com/user-attachments/assets/4b6fbbd2-33d2-4782-b1c5-99c87f898296)

Figure 3: Validation of assumptions on the toy model of learning a single attention head. (a) Assumption 4.4: weights perturbed at a random time during training (solid lines) tend back to the near-stationary point (dashed lines). (b) Assumption 4.5: weights perturbed at the beginning of a stage (solid lines) have same nonlinear evolution a

In Figure 4 The stable rank of both 
$Δ𝑊_K 𝑊_V^⊤$.
  and 
$Δ𝑊_𝑉 𝑊_𝑂^⊤$.
  increases monotonically during training, indicating that the learned representations become increasingly complex as the model trains.
The increase in stable rank reflects the ability of the model to capture richer and more diverse relationships in the data.
Differences Between 
$Δ𝑊_K 𝑊_V^⊤$.
  and 
$Δ𝑊_𝑉 𝑊_𝑂^⊤$.
The stable rank of 
$Δ𝑊_𝑉 𝑊_𝑂^⊤$.
  (orange) is consistently higher than 
$Δ𝑊_K 𝑊_V^⊤$. (blue) across all models and datasets.
This suggests that the value-output interactions are more complex and higher-dimensional compared to the key-query interactions.
Dataset Complexity:
As the dataset complexity increases (CIFAR-10 < CIFAR-100 ), the stable rank values also increase, reflecting the need for higher complexity in learned representations to handle more complex data.
We are able to repl'cate the results/

![image](https://github.com/user-attachments/assets/6711ba82-448b-494f-8530-89eee3c7c702)
a)CIFAR10
![image](https://github.com/user-attachments/assets/7b961fe1-55d7-4c1b-af51-5fe02ba7bcb5)
b)CIFAR100

Figure 4: Stable rank of $∆W_KW^⊤_Q$ (blue) and $∆W_V W^⊤_O$
(orange) on an arbitrary chosen layer
throughout training for four different pairs of networks and tasks. The stable rank of a matrix W
is defined as $∥W∥^2_F/∥W∥^2_2$, and gives a smooth approximation of the rank. Mean and standard
deviation (shaded area) are computed ac

Figure 5 visualizes the eigenvalue spectra of the weight perturbations 
$Δ𝑊_K 𝑊_V^⊤$  compared to their initial state 
$𝑊_K 𝑊_V^⊤$ at initialization across different layers in a Vision Transformer (ViT) trained on CIFAR-10. The purpose is to demonstrate the emergence of a low-rank bias in the learned weight perturbations during training.
We can"t observe similar patterns to paper.
![image](https://github.com/user-attachments/assets/12e6b657-1c68-41d3-81e2-c925d111382e)
Figure 5: Spectrum of the weight perturbation ∆WKW⊤
Q vs. initialization in a vision transformer
trained on CIFAR-10, using Adam and default initialization scale, in random self-attention heads
in different layers. The learned perturbation exhibits extreme low-rank bias post-training even in
default initialization scales. 

Figure 6 illustrates the evolution of the eigenvalues of the perturbations 
$Δ𝑊_K 𝑊_V^⊤$ and 
$Δ𝑊_𝑉 𝑊_𝑂^⊤$  during the training of a Vision Transformer (ViT) on CIFAR-10, under different initialization scales. The focus is on a single random attention head in Layer 2 throughout the training process.
We somehow replicated the graphs but their evolution has more jagged patterns.
![image](https://github.com/user-attachments/assets/bf5db9ec-9c80-4515-9cc9-517528007cd4)
Figure 6: Training a vision transformer on CIFAR-10 using Adam, while varying the initialization
scale (unit scale indicates default initialization). Plotted are the evolution of the eigenvalues of
∆WKW⊤Q (a) - (c) and ∆WV W⊤O
(d) - (f) in a random self-attention head in the second layer
throughout training. Incremental learning dynamics and a low-rank bias are evident for all scales,
albeit more pronounced at smaller initialization scales

Figure 7 presents the stable rank of 
$Δ𝑊_K 𝑊_V^⊤$  for different initialization scales in Vision Transformer (ViT) layers (Layer 1, 3, and 5). The stable rank is averaged across 8 self-attention heads per layer and highlights how the initialization scale influences the low-rank structure of the learned weights post-training.
Low-Rank Bias and Initialization Scale: Smaller initialization scales lead to a stronger low-rank bias, where the model learns simpler, more focused representations in each attention head.
Larger initialization scales result in a weaker low-rank bias, allowing for more complex and higher-rank representations.
Layer-Wise Dynamics: Deeper layers (e.g., Layer 5) require more complex representations, as evidenced by their higher stable rank compared to shallower layers (e.g., Layer 1).
This reflects the hierarchical nature of Vision Transformers, where deeper layers capture more abstract and diverse relationships.
Practical Insights for Initialization:Smaller initialization scales may be more effective in scenarios where low-rank approximations are beneficial, such as model compression or tasks requiring efficient learning.
However, overly small initialization scales may limit the representational capacity of deeper layers, especially in more complex tasks.
We are able to see this in our replication.
![image](https://github.com/user-attachments/assets/8c8417e3-3c48-49c6-9b5c-8e607b9dd2a1)
Figure 7: Stable rank of ∆WKW⊤
Q per initialization scale (Unit scale refers to the default initialization) in different self-attention heads post-training, at layers 1, 3, 5. At each layer, the stable rank
mean and standard deviation are computed across 8 heads per layer, for each initialization scale.
All models were trained on CIFAR-10 using the Adam optimizer. Smaller initialization scales lead
to lower-rank attention heads.


# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

#### <a id="1">[1]</a> Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, Lora: Low-rank adaptation of large language models, arXiv preprint arXiv:2106.09685 (2021).
#### <a id="2">[2]</a> Francis Bach, Effortless optimization through gradient flows, Machine Learning Research Blog. https://francisbach. com/gradient-flows (2020). 
#### <a id="3">[3]</a> Arthur Jacot, Francois Gaston Ged, Berfin Simsek, Cl´ement Hongler, and Franck Gabriel, Saddle-to-saddle dynamics in deep linear networks: Small initialization training, symmetry, and sparsity, 2021.
#### <a id="4">[4]</a> Jiawei Zhao, Yifei Zhang, Beidi Chen, Florian Sch¨afer, and Anima Anandkumar, Inrank: Incremental low-rank learning, arXiv preprint arXiv:2306.11250 (2023).

# Contact

Nurşen Töre email:nursen.tore@metu.edu.tr
