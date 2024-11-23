# @TODO: Paper title

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and our interpretation

## 2.1. The original method

The method, named Greg+, consists of two main parts. In contrast to state-of-the-art methods which only focus on the value of the score function, Greg+ proposes a new regularization term added to the loss, obtained by regularizing the gradient of the score function. In addition, a novel energy-based sampling method is proposed to select more informative examples from the dataset during training, which is expecially important when the auxiliary dataset is large.

## 2.1.1. Gradient Regularization

The main intuition behind gradient regularization is that the distribution of ID (in distribution) and OOD samples tend to be well-separated, and the area around an ID sample would not include any OOD samples, and vice versa.

![t-SNE plot showing the representation of ID and OOD datasets for CIFAR experiments](./Figures/Distribution.png)

**Figure 1: t-SNE plot showing the representation of ID and OOD datasets for CIFAR experiments**

Regularization term is used to promote smoothness of the loss function around the training samples by penalizing the norm of its gradient. For the first-order Taylor approximation of the score function $S(x)$,

![First-order Taylor Approximation of the Score Function](./Figures/ScoreFunctionApproximation.png)

Suppose $x$ is a training sample, and $x$' is another sample which is sufficiently close to $x$. If the gradient $\Delta_x S(x)$ is small, then $|S(x) - S(x')|$ becomes small for nearby points $x$'. This ensures ID points classified as ID remain stable within their neighborhood.
As the paper mainly focuses on the energy loss as the score function, $L_{\Delta S}$ is defined as follows:

![Gradient of the energy loss](./Figures/GradientEnergyLoss.png)

Thresholds $m_{in}$ and $m_{aux}$ are selected such that the loss only penalizes the gradient for the correctly detected samples. For a sample $x$, if $S(x)$ is below the threshold, we label it as ID, and label it as OOD otherwise.
Following two cases could be considered to better understand the intuition behind this choice:
1. If an ID (or OOD) sample is correctly detected, penalizing the gradient around that sample reduces sensitivity to small changes in the input, providing stable ID (OOD) classification.
2. If a sample is misclassified, gradient regularization is not applied.

The models mentioned in the experiments are trained using the following loss,

**$L = L_{CE} + \lambda_S L_S + \lambda_{\Delta S} L_{\Delta S}$**

where $\lambda_S$ and $\lambda_{\Delta S}$ are regularization hyperparameters and $L_{CE}$ is the well-known cross-entropy loss.

## 2.1.2. OOD Sampling

Auxiliary OOD datasets may be larger than the ID dataset, and using all OOD samples during training may be computationally expensive and lead to bias towards specific regions of the feature space. To address this issue, a novel energy-based sampling method is proposed in the paper. Steps of the proposed sampling method can be explained as follows:
1. For each auxiliary OOD sample $x$, we first calculate the features $z = h(x)$, where $h(.)$ is the feature extractor. Then, energy score $s = -LSE(f(x))$ is calculated, where $f(x)$ is the logit. As features with larger magnitudes may lead to biased clustering, the feature vectors z are normalized as follows:

$\t$ $z = \frac{z}{||z||}$

2. a

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
