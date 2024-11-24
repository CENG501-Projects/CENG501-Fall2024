# FIGHTING OVER-FITTING WITH QUANTIZATION FOR LEARNING DEEP NEURAL NETWORKS ON NOISY LABELS

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

An increase of the computation power let to emergence of computationally complex neural networks. These neural networks require large amount of annotated data. Collecting such amount of annotated data without noisy labels are very costly. Therefore, techniques to cope with noisy labeled annotated data are required. In this paper, it is suggested that by restricting the expressivity of the neural network, accuracy of the network that is trained with noisy labeled data can be increased. The expressivity of the neural network can be restricted by using classical regularization and compression techniques. It is claimed that these techniques are not tested with noisy labels.

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and our interpretation

## 2.1. The original method

In this paper, 2 methods were introduced to fight over-fitting in deep neural networks. The first one is regularization techniques and the second one is compression techniques. The authors decided to use early stopping, weight decay, dropout, and label smoothing methods as regularization techniques and pruning and quantiziation methods as compression techniques. 

<h3>2.1.1. Regularization Techniques</h3>

This techniques are used by improving the generalization capability of the model. In order to get better results in test set without chaging the implementation, we can use these methods on training and validation sets. Let's define a neural network F with L layers f <sub>l</sub>, weights W<sub>l</sub>. 

<h4>2.1.1.1. Early Stopping</h4>
During the training of F (neural network), the accuracy initially improves on both the training and validation datasets. However, overfitting occurs when the accuracy on the validation and test datasets starts to decline, even though it continues to rise on the training set. This approach is used on the training and validation datasets to identify the best point to stop training the model.

<h4>2.1.1.2. Weight Decay</h4>
This method helps prevent overfitting by imposing a constraint on the scale of all weight values in F. Specifically, it involves adding an L2 regularization term to the training loss, defined as: 

**L<sub>w</sub> = α<sub>w</sub> ∑<sub>l=1</sub><sup>L</sup> ‖W<sub>l</sub>‖<sub>2</sub><sup>2</sup>**

where **1/α<sub>w</sub>** determines the scale enforced on the weights.

<h4>2.1.1.3. Dropout </h4>
Dropout helps prevent overfitting by randomly deactivating parts of **F**, encouraging the model to make predictions using smaller sub-networks and avoiding weight co-adaptation. During training, each scalar weight is set to 0 with a probability of **p** for every example. At test time, all weights are scaled by **p** to reflect their activation frequency during training.

<h4>2.1.1.4. Label Smoothing </h4>
This method helps reduce overfitting by preventing neural networks from becoming overly confident in their predictions. It achieves this by modifying the ground truth labels as follows:

**y<sub>α<sub>s</sub></sub> = (1 − α<sub>s</sub>)y + α<sub>s</sub> * (1 / C)**

Here, **α<sub>s</sub>** controls the smoothing intensity, and **C** represents the number of classes.

In multi-task binary classification tasks, like action unit (AU) detection, label smoothing is applied separately to each label, using **C = 2** to represent the presence or absence of a label.

It’s important to highlight that these methods do not impact the network's inference runtime, leaving the efficiency problem unaddressed. As a result, the authors suggest investigating compression techniques as a potential solution for regularization.

<h3>2.1.2. Compression Techniques</h3>
<h4>2.1.2.1. Pruning </h4>
This method works on the assumption that **F** is already pre-trained. For each **f<sub>l</sub>**, it applies standard magnitude-based structured pruning to the weight tensors **W<sub>l</sub>**, removing neurons with the largest L1 norm.

The idea behind this approach is that smaller weights lead to smaller activations, which have less influence on the decision-making process. By limiting the model's complexity, pruning reduces the likelihood of overfitting.

<h4>2.1.2.2. Quantization </h4>
### Quantization

The standard quantization operator, in **b** bits, is defined as:

**quantized(X) = ⌊X \* (2<sup>b−1</sup> − 1) / λ<sub>X</sub>⌉**

where **⌊·⌉** denotes rounding to the nearest integer, and **λ<sub>X</sub>** is a scaling parameter specific to **X**. This parameter ensures that the range of **X** is correctly mapped to **[−(2<sup>b−1</sup> − 1); 2<sup>b−1</sup> − 1]**.

In practice, scalar values are typically used for **λ<sub>X</sub>** when quantizing activations (e.g., layer inputs), while vector values are used for weight tensors (per-channel quantization). The scaling parameter for activations is estimated per batch during training as:

**λ<sub>X</sub> = max{|X|}**

An exponential moving average is then used to update the inference value. Conversely, weight scales are always computed as the maximum value per channel of **|W<sub>l</sub>|**.

When optimizing the weight values W, the rounding operator creates zero gradients in most places, making gradient-based optimization difficult. To solve this, the **straight-through estimation (STE)** method is used, which replaces the gradient of the quantization process with an identity function. This method also removes batch-normalization layers from the network architecture.


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

1. G. Tallec, E. Yvinec, A. Dapogny, and K. Bailly, "Fighting over-fitting with quantization for learning deep neural networks on noisy labels," arXiv, 2023. [Online]. Available: https://arxiv.org/abs/2303.11803.

2. A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," in Proceedings of the Semantic Scholar Corpus, 2009. [Online]. Available: https://api.semanticscholar.org/CorpusID:18268744.

3. X. Zhang, L. Yin, J. F. Cohn, S. Canavan, M. Reale, A. Horowitz, P. Liu, and J. M. Girard, "BP4D-Spontaneous: A High-Resolution Spontaneous 3D Dynamic Facial Expression Database," in Image and Vision Computing, vol. 32, no. 10, pp. 692-706, 2014. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0262885614001012.

4. X. Zhang, L. Yin, J. F. Cohn, S. Canavan, M. Reale, and A. Horowitz, "A High-Resolution Spontaneous 3D Dynamic Facial Expression Database," in Proceedings of the 10th IEEE International Conference and Workshops on Automatic Face and Gesture Recognition (FG), 2013, pp. 1-6. [Online]. Available: https://doi.org/10.1109/FG.2013.6553788.

# Contact
Yüksel Pelin Kılıç - pelinkilic97@gmail.com
Hakan Çakmak - cakmakhakan.boun@gmail.com
