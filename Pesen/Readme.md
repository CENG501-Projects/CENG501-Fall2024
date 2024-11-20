# Entropy Induced Pruning Framework for Convolutional Neural Networks

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper "Entropy Induced Pruning Framework for Convolutional Neural Networks" by Yiheng Lu et al. was presented at the 38th AAAI Conference on Artificial Intelligence (AAAI-2024). This work addresses the challenge of efficient pruning in convolutional neural networks (CNNs) and proposes a novel pruning framework, Average Filter Information Entropy (AFIE).

Pruning techniques are essential for reducing the computational overhead of CNNs, enabling their deployment in resource-constrained environments like mobile devices and embedded systems. The AFIE method evaluates the importance of filters in a CNN using entropy derived from the eigenvalues of the layer's weight matrix, allowing effective pruning even when the model is under-trained.

###Project Goal:
The primary goal of this project is to reproduce the results presented in the paper to validate its claims. This involves:

1. Implementing the AFIE-based pruning framework.
2. Testing its performance on AlexNet, VGG-16, and ResNet-50 models using datasets such as MNIST, CIFAR-10, and ImageNet.
3. Comparing the experimental outcomes with the original results in terms of parameter reduction, computational savings (FLOPs), and accuracy recovery.

By conducting this reproducibility study, we aim to evaluate the practicality of the proposed method and explore its implications in relation to the existing pruning literature.


## 1.1. Paper summary

CNNs have achieved remarkable success in tasks like image classification but are computationally intensive due to their large model size. Redundancy in CNN filters not only increases inference time and memory requirements but also risks overfitting. Existing pruning techniques can be broadly categorized into:

1. **Unstructured Pruning**: Removes individual weights but retains overall model structure.
2. **Structured Pruning**: Removes entire filters or layers, significantly reducing computational demands.
Structured pruning is preferred for real-world applications as it achieves better computational efficiency by reducing the number of feature maps.

However, current methods depend on fully trained models to assess the importance of filters accurately. This dependency makes pruning inefficient and resource-intensive, as under-trained models often yield suboptimal results.

### Proposed Method:
The paper introduces Average Filter Information Entropy (AFIE), a structured pruning framework that eliminates the need for a fully trained model by evaluating filter importance using entropy derived from the eigenvalues of the weight matrix. AFIE's key steps include:

**Weight Decomposition**: Applying Singular Value Decomposition (SVD) to decompose the weight matrix into a low-rank space.  
**Normalization and Entropy Calculation**: Eigenvalues are normalized into a probability distribution, and entropy is calculated to measure redundancy.
**Pruning Ratio Allocation**: Based on AFIE scores, pruning ratios are dynamically allocated across layers.
#### Key Contributions:

1. The method eliminates the reliance on fully trained models, allowing effective pruning even when the model is trained for just one epoch.
2. The AFIE framework provides consistent filter evaluations regardless of the model's training stage.
3. The authors demonstrate competitive results on AlexNet, VGG-16, and ResNet-50 across datasets, achieving significant reductions in parameters and FLOPs while maintaining comparable accuracy.

### Relation to Existing Literature:
AFIE addresses limitations in both *Layer-Importance-Supported (LIS)* and *Filter-Importance-Supported (FIS)* pruning methods, which often rely on weight magnitude, gradients, or sensitivity analysis. The proposed entropy-based approach is less dependent on model optimization, making it a robust alternative to traditional methods like Taylor expansion, L1 norm, and batch normalization regularization.

This framework bridges the gap between pruning efficiency and model training, contributing to the broader goal of making deep learning models more accessible and efficient.

# 2. The method and our interpretation

## 2.1. The original method

@TODO: Explain the original method.

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

1. Paul Gavrikov, Towards Data Science (2022, October 28). Downloading and Using the ImageNet Dataset with PyTorch. Train your image classification models with the most popular research dataset. https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
