# Viewing Transformers Through Lens of Long Convolutions Layers

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

Transformers have been dominating Deep Learning areas, especially NLP domains, for years after their foundation. However, they perform poorly on longe range tasks and struggle to exploit long context compared to long range layers, such as state-space layers, linear RNN layers and global convolution layers.

In this paper [1] (ICML 2024), the authors identify the principles of long range layers that allow them to capture long range relations. They also discuss the possible reasons behind tranformers' sub-optimal performance in these tasks. Building on this analysis, they propose Local and Smooth Attention (LaS-Attention), a simple modification to the vanilla transformer architecture that improves its ability to handle long-range relationships. This modification leads to performance enhancement on the Long Range Arena (LRA) benchmark.

This repository aims to reproduce the results indicated in the paper.

## 1.1. Paper summary

### Summary

This paper investigates the sub-optimal performance of transformers on long-range tasks in terms of expressiveness, optimization and generalization.

**(i) Expressiveness.** Since transformers are high-capacity models, this is unlikely to be a cause of the problem. Furthermore, it is proven in the appendix of the paper that one head self-attention can express one channel of the state-space layer.

**(ii) Optimization.** This paper associates optimization issues for long-range dependencies with exploding and vanishing gradient problems. However, this is not the primary bottleneck in transformers for three reasons. Firstly, since self-attention heads are parallel, there is no reason to assume that gradients are more likely to vanish or explode on long interactions. Secondly, the amount of nonlinearity is constant in transformers. Thirdly, trasformers extensively use normalization layers which makes them stable.

**(iii) Generalization.** The lack of generalization due to an unsuitable inductive bias that results in an unfavorable hypothesis class is likely to be the root cause of the problem. When the models exhibiting exceptional performance on LRA benchmarks are examined, it is seen that they tend to contain layers with
strong inductive bias. Furthermore, the results of the paper shows a significant improvement in the performance of proposed models on the LRA benchmark with increasing amount of data. The same phenomenon is not observed in vanilla transformer architecture. This highlights the fact that the model’s ability to fit the underlying data distribution increases with the right type of inductive bias.

### Contribution to Existing Literature

This paper explores why transformers struggle with tasks that involve long-range dependencies and identifies key principles—like smoothness and locality—that help models handle these tasks better. The authors introduce Local and Smooth Attention (LaS-Attention), a simple modification to transformers that incorporates these principles by smoothing attention scores and adding a positional bias to focus on nearby tokens. Unlike other approaches such as state space layers, it doesn’t rely on complex 1-D convolution operations but still performs very well on the LRA benchmark. The paper bridges the gap between transformers and models designed for long-range tasks. It also introduces LaS-chunk, which is a linear complexity solution to the same problem.

# 2. The method and our interpretation

## 2.1. The original method
Local and Smooth (LaS) Attention exploits the principles of smoothness and exponentially decaying structure, which can be observed in the following definition of the $c^{th}$ LaS attention head calculation: 

$$ LAS_c(Q,K,V) = AP\left(SF\left(exp\left(-\alpha_c D_L\right) \odot \left(\frac{QK^T}{\sqrt{d_k}}\right)\right)\right)$$

Architecture of LaS attention can be seen in figure below.

![image](https://github.com/user-attachments/assets/d5aa4895-da99-4186-b50f-d22249d48da2)



### The Principle of Smoothness
LaS Attention exploits this principle by a smoothing operator implemented by 1-D average pooling (denoted by $AP()$ in the above formula) applied to each row individually with appropriate padding to preserve the shape.

### The Principle of Exponentially Decaying Structure
LaS Attention exploits this principle by elementwise multiplication of the attention matrix at each head with a nonlearnable locally decaying matrix. This is achieved by Exponentially Locally Decay (ELD) operator. This operator is defined by

$$ ELD: \mathbb{R}^{LxL} \rightarrow \mathbb{R}^{LxL} $$

$$ ELD(B) = exp\left(-\alpha_c D_L\right) \odot B $$ 

where the ELD matrix is defined as

$$ ELD = exp\left(-\alpha_c D_L\right) $$

$D_L$ is the distance matrix multiplied by the causality mask ($-\alpha_c$). The distance matrix is computed as follows:

![image](https://github.com/user-attachments/assets/ee1cb4c7-2290-4c8c-b4de-cc53777bd0d5)


LaS Attention utilizes different $\alpha_c$ values for each attention head to allow each attention head to focus on dependencies of a uniform scale. As a result of this application, the model can capture a spectrum of local dependencies at multiple scales at each layer. This creates a hierarchy between local interactions, allowing the recognition of global dependencies.

Initialization of $\alpha_c$ is realized as follows:

**(i)** $\alpha_0$ is set to 0 in first attention head.

**(ii)** $\alpha_c$ initialized exponential-uniformly in $[0,B]$, where $B$ is defined as a hyperparameter in (0,1).

## 2.2. Our interpretation

Below are some of our interpretations about aspects that were unclear in the paper:

**(i)** The paper does not explicitly state whether positional encoding is used as it is in vanilla transformer architecture. We inferred that it is not included in the model since the Exponentially Locally Decay (ELD) already captures positional information as indicated in [2].

**(ii)** Since the padding value is not specified, we assumed 0-padding as it is the default padding used by torch.nn.AvgPool1d().

**(iii)** The paper doesn't explain why exponential function is used in ELD. We inferred that it is likely because the exponential decay ensures non-negativity and smooth, continuous transition of influence on attention scores as $D_L$ changes.

**(iv)** The authors of the paper build their repository upon the existing S4 repository. The S4 repository has 2 configuration files for LRA tasks, one is older, and the other is more recent. There are slight differences between them. The most important differences are seed and learning rate schedule. The value of the seed changes for different LRA tasks. The old version uses a plateau-based scheduler, ReduceLROnPlateau, and the newer version employs a cosine schedule with warm up. We preferred to use the plateau-based scheduler since it is adaptive, resource-efficient and driven by training performance, which makes it suitable for our situation. Authors state in the paper that they obtained their results by averaging over three different seed values; however, we do not have enough computational power to conduct our experiments in that way. Hence, we used seed=1112 in all of our experiments

**(v)** By observing the figure below, we inferred that as the $\alpha_c$ value increases, the weights corresponding to distant neighbours approaches to 0. This puts more emphasis on close neighbours. 

<img width="861" alt="image" src="https://github.com/user-attachments/assets/e929a3b0-608b-4fb7-a050-fb70edd0782a">



# 3. Experiments and results

## 3.1. Dataset

There are five distinct experiments conducted on five tasks of LRA Benchmark. Therefore, we begin to explain experiments by first introducing the LRA benchmark and the tasks under this benchmark.

According to [3], the LRA benchmark is a systematic framework designed to evaluate the performance of Transformer models in long-context scenarios. It includes tasks for testing the ability to handle sequences ranging from 1,000 to 16,000 tokens across various modalities like text, image and spatial reasoning. The five tasks of LRA are **ListOps**, **Text Classification**, **Document Retrieval**, **Image Classification** and **Path Finder**. There is also a task named **Path Finder-X**, extending Path Finder with extreme lengths. This task is not included in the performance evaluation of Las Attention. Therefore, we suffice to explain the five of them.

**(i) ListOps:** The dataset consists of sequences with a hierarchical structure and operators MAX, MEAN, MEDIAN and SUM_MOD that are enclosed by delimiters. The model needs to access all tokens and model the logical structure of the inputs in order to make a prediction.

**(ii) Text Classification:** The dataset consists of text sequences at the byte or character level. The model needs to reason with compositional, unsegmented data in order to solve a meaningful real-world task.

**(iii) Document Retrieval:** The dataset consists of document pairs represented at the byte or character level. The model needs to compress long sequences into representations suitable for similarity-based matching.

**(iv) Image Classification:** The dataset consists of images represented as sequences of pixels. The model needs to learn the 2D spatial relations between input pixels.

**(v) Path Finder:** In this task, the model needs to make a binary classification indicating whether two points are connected with a by a path.

The performance of Las Attention is evaluated based on these tasks. We evaluated our implementation of LaS Attention on ListOps and Document Retrieval tasks only due to our limited time and resources.


## 3.2. Experimental setup

The original paper provides the following hyperparameters for the experimental setups for different LRA tasks:

Table 1: Hyperparameter Configurations
<img width="900" alt="LRA parameters" src="https://github.com/user-attachments/assets/ad5f033b-7d65-49b5-af5a-d59dce0dd289">

In the Table 1, LR is learning rate and WD is weight decay. BN and LN refer to Batch Normalization and Layer Normalization, respectively.

The hyperparameters for the LaS attention are as follows:

**(i)** B, controlling $\alpha_c$

**(ii)** the 1-D average pooling window size P

The authors built their experiments upon the existing S4 repository. In all experiments, they used causal transformers with 8 heads, and aligned training procedures and hyperparameters with the S4 repository. In another words, they used the default hyperparameters used in the S4 repository for the ones that are not specified in the above table. Dropout was set to 0 in all cases.

We conducted our experiments on the ListOps and Document Retrieval tasks, as indicated in Section 3.1 of the paper, using a Google Colab instance with an A100 GPU. While Google Colab provides 40GB of GPU RAM for this GPU type, we had to reduce the batch sizes to avoid exceeding the memory limit for both tasks. Additionally, we were unable to complete the full number of epochs specified in the paper due to resource constraints. Apart from these adjustments, we used the same hyperparameters and architecture settings described in the paper. Lastly, we utilized the datasets in their full size without any reductions in the training, test or validation sets.

The hyperparameters we adjusted to suit our setup can be observed in Table2. The number of epochs of Document Retrival task is considerably reduced compared to the setting of original paper. This is due to the fact that one epoch takes 3.5 hours with full dataset.

Table 2: Batch size and epoch adjustments in our experiments
|                     | Batch Size | Epoch|
|---------------------|--------|----------|
| **Listops** |        | 40 |
| **Document Retrieval** |    6    | 7 | 

-------------------------

We ran our first experiment on Sequential MNIST (sMNIST); however, the paper does not provide setup details for this. For this reason, we used the setup parameters of the LRA Image task with the exception of batch size and number of epochs. We reduced the batch size to 10 to not exceed the Google Colab T4 GPU RAM. We also reduced the number of epochs due to limited runtime of Google Calob. Similar to the original paper, we built our code upon the existing S4 repository. Also, we adapted the Transformer architecture from [3] to hanve more flexibility in our architecture.

We conducted two experiments:

**(i)** First experiment uses the whole dataset and the model is trained for two epochs.

**(ii)** Second experiment uses 1/20 of the dataset and the model is trained for 80 epochs.


## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

**1st Experiment (Full sMNIST Dataset and 2 Epochs):**

Figure 1: Train and Validation loss for sMNIST   

![image](https://github.com/user-attachments/assets/fdb9e367-402d-4029-ba85-b49bcb20a4da)


**2nd Experiment (1/20 sMNIST Dataset and 80 Epochs):**

Figure 2: Train and Validation loss for small sMNIST 

![image(2)](https://github.com/user-attachments/assets/d34e6291-61c8-41d5-bae4-e7c61f1da04f)


Table 2: Training and validation accuracies for our implementation
|                     | Small (1/20) sMNIST | Full sMNIST|
|---------------------|--------|----------|
| **Training** |    13.66    | 10.348 |
| **Validation** |    13.60    | 8.633 | 

<br><br>

Figure 3: Original paper's accuracy results  

<img width="350" alt="Papers_MNIST_results" src="https://github.com/user-attachments/assets/ad2ce844-7860-4b1b-9871-3cfcb8541344">

<br><br>

**Discussion:**  

Our accuracy was much lower compared to the original paper. This could be because we used the setup for the LRA Image task, while the original paper might have used different settings for the sMNIST task. A smaller batch size might have also impacted the accuracy. It's possible that we made a mistake in processing or interpreting the model's output. This might include errors in how predictions were extracted, how metrics were calculated, or how data was handled in post-processing. Lastly, we couldn’t finish the first experiment with the full dataset because of Google Colab's runtime limits. All of them might be a reason of our low accuracy.

**3rd Experiment (Full LRA Listops Dataset, 40 Epochs):**

Figure 4: Train Loss and Accuracy for LRA Listops Benchmark

<img width="650" alt="train_graphs" src="https://github.com/user-attachments/assets/c4008269-8bdc-4f04-a028-4b49536bf526" />


Figure 5: Validation Loss and Accuracy for LRA Listops Benchmark

<img width="650" alt="val_graphs" src="https://github.com/user-attachments/assets/2b2088cd-229f-4a45-800e-455a2841c9a8" />


Figure 6: Test Loss and Accuracy for LRA Listops Benchmark

<img width="650" alt="test_graphs" src="https://github.com/user-attachments/assets/c0fbd276-7307-41c7-ad7b-6644cd5b29ea" />




Figure ??: Original Paper's Accuracy Results for LRA Benchmarks

<img width="650" alt="papers_results" src="https://github.com/user-attachments/assets/449671e3-f5a9-47a5-925b-ba08a80eef8a" />

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

[1] Zimerman, I., & Wolf, L. (2024). Viewing Transformers Through the Lens of Long Convolutions Layers. Proceedings of Machine Learning Research, 235, 62815-62831.  
[2] Press, O., Smith, N. A., & Lewis, M. (2021). Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409. 
[3] Tay, Yi, et al. "Long range arena: A benchmark for efficient transformers." arXiv preprint arXiv:2011.04006 (2020). 
[4] https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master

# Contact

Name: Defne Ekin Email: ekindefne@gmail.com  
Name: Şevval Uçar Email: seevvalucar@gmail.com
