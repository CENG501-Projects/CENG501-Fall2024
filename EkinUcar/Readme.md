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

![image](https://github.com/user-attachments/assets/275696b6-2222-4dcf-8eef-5268e4fe3b0c)


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

### The Principle of Smoothness

### The Principle of Exponentially Decaying Structure

The paper does not explicitly state whether positional encoding used similarly to vanilla transfermer architecture

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
[1] OUR PAPER
[2] ALIBI

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
