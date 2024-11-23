# @TODO: Paper title

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

Enhancing the Power of OOD Detection via Sample-Aware Model Selection is a paper authored by Feng Xue et al., presented at CVPR 2024. This work introduces ZODE (Zoo-based OOD Detection Enhancement), a novel approach aimed at improving out-of-distribution (OOD) detection by leveraging sample-aware model selection within a model zoo framework. The method advances the field of OOD detection, a critical task for ensuring the reliability and safety of machine learning systems in open-world environments.

The primary goal of this project is to reproduce the results and insights presented in the paper. This includes implementing ZODE from scratch, verifying its performance, and analyzing its contributions in the context of existing literature. By achieving reproducibility, this effort aims to validate the claims made in the paper and potentially extend its applicability.

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).

## 1.1. Paper summary

The paper “Enhancing the Power of OOD Detection via Sample-Aware Model Selection” introduces ZODE (Zoo-based OOD Detection Enhancement), a framework that improves out-of-distribution (OOD) detection by leveraging a diverse collection of pre-trained models, termed a "model zoo." Unlike traditional methods that rely on a single pre-trained model, ZODE dynamically selects models based on each test sample. This sample-aware model selection enables more robust and accurate OOD detection.

ZODE uses statistical techniques to normalize detection scores into p-values and adjusts thresholds via the Benjamini-Hochberg procedure, ensuring a high true positive rate (TPR) for in-distribution samples while reducing false positives. Theoretical analysis proves ZODE's ability to maintain TPR and asymptotically lower false positive rates (FPR) as the model zoo grows.

Extensive experiments on CIFAR10 and ImageNet demonstrate ZODE's effectiveness, achieving a 65.40% improvement in FPR on CIFAR10 and a 37.25% improvement on ImageNet compared to the best baselines. The paper highlights ZODE’s ability to exploit the complementarity of multiple models while addressing the limitations of single-model detectors. However, the method requires significant storage and computational resources, which could be mitigated through distributed computing.

In summary, ZODE offers a novel, statistically grounded approach to OOD detection that combines theoretical guarantees with state-of-the-art empirical performance, making it a significant advancement in the field.

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and our interpretation

## 2.1. The original method

The paper proposes ZODE (Zoo-based OOD Detection Enhancement), which improves out-of-distribution (OOD) detection by leveraging a diverse set of pre-trained models (a "model zoo") and applying sample-aware model selection. The key steps of the method are:

- Model Zoo Construction: A collection of pre-trained models is built with diverse architectures and training strategies, ensuring the ability to capture a wide range of features.

- P-Value Normalization: The OOD detection scores for a given test input are transformed into p-values for each model in the zoo. The p-value represents the probability of the sample belonging to the in-distribution based on the model's scoring function.

- Threshold Adjustment: To maintain a high true positive rate (TPR) for in-distribution samples while reducing false positive rates (FPR), the paper uses the Benjamini-Hochberg procedure. This adjusts the detection threshold dynamically, ensuring statistical rigor across the model zoo.

- Sample-Aware Selection: For each test input, the method determines the subset of models that classify the input as OOD. If no models in the zoo classify the input as OOD, it is considered in-distribution; otherwise, it is flagged as OOD.

The method is backed by theoretical analysis, proving that ZODE maintains TPR while achieving low FPR as the zoo size increases. Empirically, ZODE is shown to leverage the complementarity of models, significantly outperforming single-model detectors and naïve ensembles.

@TODO: Explain the original method.

## 2.2. Our interpretation

The ZODE method strikes a balance between statistical rigor and practical applicability. The transformation of detection scores into p-values simplifies the challenge of comparing outputs from heterogeneous models, which we find to be an elegant solution to standardizing scores. The use of the Benjamini-Hochberg procedure is particularly impressive, as it avoids the accumulation of errors typically seen in naïve ensembles, while ensuring statistical consistency.

From our perspective, the sample-aware model selection mechanism is a powerful innovation. It allows the detection process to adapt dynamically to each test input, leveraging the strengths of different models in the zoo. However, we recognize potential challenges in real-world implementations. The need to store validation data for p-value computation and to perform per-sample model selection may result in significant computational and memory overhead, particularly for large-scale deployments. A distributed approach might address these issues, though it could add implementation complexity.

We believe ZODE’s success hinges on the quality and diversity of the model zoo. While this is a strength, it may also present a barrier for smaller teams or projects with limited resources. Despite these challenges, we see ZODE as a transformative approach that combines theoretical robustness with practical effectiveness, making it a significant step forward in OOD detection research.

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

The authors of the paper evaluated ZODE on two main datasets: CIFAR10 and ImageNet-1K, representing in-distribution (ID) data, with various out-of-distribution (OOD) datasets used for evaluation.

### Datasets:
- CIFAR10 (ID) and six OOD datasets: SVHN, LSUN, iSUN, Texture, Places365, and CIFAR100.
- ImageNet-1K (ID) and four OOD datasets: subsets of Places365, iNaturalist, SUN, and Texture.

### Metrics:
The experiments were evaluated using:
- True Positive Rate (TPR) for ID samples.
- False Positive Rate (FPR) for OOD samples at a TPR of 95%.
- Area Under the ROC Curve (AUC), capturing overall performance across varying thresholds.

### Model Zoo:
- CIFAR Experiments: The zoo consisted of seven pre-trained models (e.g., ResNet18, ResNet50, DenseNet, etc.), varying in architecture and loss functions (including contrastive loss for diversity).
- ImageNet Experiments: A larger zoo with diverse architectures, including ResNet50*, ResNeXt101, Swin Transformer models, and DinoV2, pre-trained on different datasets with varying resolutions.

### Baseline Methods:
The paper compared ZODE against state-of-the-art OOD detectors, including methods like Maximum Softmax Probability (MSP), Energy-based models, KNN, and Mahalanobis distance.

Thresholds and Hyperparameters:
- ZODE used the Benjamini-Hochberg procedure to adjust detection thresholds dynamically.
- Key hyperparameters like significance levels for p-values (αα) were tuned to maintain a TPR of 95% for ID samples.

### Our Adjustments

For our implementation, we plan to:

- Maintain the same datasets and baselines as described in the original paper for consistency.
- Recreate the model zoo with accessible pre-trained models from standard repositories like PyTorch or TensorFlow. However, the exact pretraining strategies (e.g., contrastive loss) might differ if specific models are unavailable.
- Use the same evaluation metrics (FPR, TPR, and AUC) to align with the original results.
- Experiment with scaling the zoo size and tuning hyperparameters (αα) to observe its impact on ZODE's performance.

These adjustments are aimed at ensuring reproducibility while accommodating practical constraints, such as the availability of pre-trained models and computational resources.

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
