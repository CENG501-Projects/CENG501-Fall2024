# Enhancing the Power of OOD Detection via Sample-Aware Model Selection

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

Enhancing the Power of OOD Detection via Sample-Aware Model Selection [1] paper, authored by Feng Xue et al., is published at the CVPR 2024 (Conference on Computer Vision and Pattern Recognition) conferance. This paper introduces ZODE (Zoo-based OOD Detection Enhancement) method, a novel approach whose purpose is to improve out-of-distribution (OOD) detection by leveraging sample-aware model selection within a model zoo framework. The method advances the field of OOD detection, which is critical for ensuring the reliability of machine learning systems in real-world cases.

The goal of this repository is to reproduce the results and insights presented in the paper. This includes implementing the ZODE method from scratch, verifying its performance with the same metrics presented in the paper. 

## 1.1. Paper summary

Out-of-distribution (OOD) detection is a critical deep learning task that identifies inputs that differ from the general distribution of the training data. While deep neural networks are sufficient in scenarios where the test and training distributions are identical or similar, their performance often decreases when supplied with OOD inputs. This limitation causes significant risks in applications where safety is crucial since undetected OOD inputs can lead to error prone decisions. 

To address this, the paper introduces ZODE, a novel OOD detection method that uses a diverse set of pre-trained models known as a model zoo. Unlike traditional methods, which use a single pre-trained model, ZODE chooses models dynamically based on each test sample. The sample-aware model selection allows for more robust and accurate OOD detection. ZODE normalizes detection scores into p-values and adjusts thresholds using the Benjamini-Hochberg procedure, resulting in a high true positive rate (TPR) for in-distribution samples while reducing false positives. Theoretical analysis demonstrates ZODE's ability to maintain TPR while asymptotically decreasing false positive rates (FPR) as the model zoo grows.

Comprehensive experiments on CIFAR10 and ImageNet datasets demonstrate ZODE's effectiveness by achieving a 65.40% improvement in FPR on CIFAR10 and a 37.25% improvement on ImageNet compared to the best baselines. The paper emphasizes how ZODE leverages the strengths of multiple models to overcome the weaknesses of single-model detectors.However, the method requires significant storage and computational resources, which could be mitigated through distributed computing.

In summary, ZODE offers a novel, statistically grounded approach to OOD detection that combines theoretical guarantees with state-of-the-art empirical performance, making it a significant advancement in the field.

# 2. The method and our interpretation

## 2.1. The original method

The paper proposes ZODE, which improves OOD detection by leveraging a collection of pre-trained models named a "model zoo" and applying sample-aware model selection. The core components of the method are:

- **Model Zoo Construction:** A collection of pre-trained models is joint together with diverse architectures and training strategies, ensuring the ability to capture a wide range of features.

- **p-Value Normalization:** The OOD detection scores for a given test input are transformed into p-values for each model in the model zoo. The p-value represents the probability of the sample belonging to the ID based on the model's scoring function.

- **Threshold Adjustment:** To maintain a high true positive rate (TPR) for ID samples while reducing false positive rates (FPR), the paper uses the Benjamini-Hochberg procedure [2]. This dynamically adjusts the detection threshold, ensuring precise and statistically robust integration across the model zoo.

- **Sample-Aware Selection:** The model zoo classifies an input as in-distribution (ID) if all models agree it is ID; otherwise, it is considered OOD. However, as the number of models increases, this naive approach is likely to accumulate errors, resulting in decreased performance.

The following algorithm outlines the ZODE framework, which dynamically selects active models from a pre-trained model zoo to robustly classify inputs as ID or OOD:
![alt text](<images/score.png>) 
Figure 1: The empirical distribution of the score function.
![alt text](<images/threshold.png>)
Figure 2: The largest subscript that satisfies the threshold condition
![alt text](<images/algorithm.png>)
Figure 3: Zoo-based OOD Detection Enhancement

Theoretical analysis backs up the approach, showing that even as the model zoo expands, ZODE continuously maintains a high TPR while maintaining a low FPR. According to empirical findings, ZODE outperforms both single-model detectors and naive ensemble approaches by efficiently utilizing the advantages of multiple models.

## 2.2. Our interpretation

The ZODE method strikes a balance between statistical rigor and practical applicability. The transformation of detection scores into p-values simplifies the challenge of comparing outputs from heterogeneous models, which we find to be an elegant solution to standardizing scores. The use of the Benjamini-Hochberg procedure is particularly impressive, as it avoids the accumulation of errors typically seen in naïve ensembles, while ensuring statistical consistency.

From our perspective, the sample-aware model selection mechanism is a powerful innovation. It allows the detection process to adapt dynamically to each test input, leveraging the strengths of different models in the zoo. However, we recognize potential challenges in real-world implementations. The need to store validation data for p-value computation and to perform per-sample model selection may result in significant computational and memory overhead, particularly for large-scale deployments. A distributed approach might address these issues, though it could add implementation complexity.

We believe ZODE’s success hinges on the quality and diversity of the model zoo. While this is a strength, it may also present a barrier for smaller teams or projects with limited resources. Despite these challenges, we see ZODE as a transformative approach that combines theoretical robustness with practical effectiveness, making it a significant step forward in OOD detection research.

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

The authors of the paper evaluated ZODE on two main datasets: CIFAR10 [3] and ImageNet-1K [4], representing in-distribution (ID) data, with various out-of-distribution (OOD) datasets used for evaluation.

### Datasets:
- CIFAR10 (ID) and six OOD datasets: SVHN [5], LSUN [6], iSUN [7], Texture [8], Places365 [9], and CIFAR100 [10].
- ImageNet-1K (ID) and four OOD datasets: subsets of Places365, iNaturalist [11], SUN [12], and Texture.

### Metrics:
The experiments were evaluated using:
- True Positive Rate (TPR) for ID samples.
- False Positive Rate (FPR) for OOD samples at a TPR of 95%.
- Area Under the ROC Curve (AUC), capturing overall performance across varying thresholds.

### Model Zoo:
- CIFAR Experiments: The zoo consisted of seven pre-trained models (e.g., ResNet18, ResNet50, DenseNet, etc.), varying in architecture and loss functions (including contrastive loss for diversity).
- ImageNet Experiments: A larger zoo with diverse architectures, including ResNet50*, ResNeXt101, Swin Transformer models, and DinoV2, pre-trained on different datasets with varying resolutions.

### Baseline Methods:
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
1. Xue, F., He, Z., Zhang, Y., Xie, C., Li, Z., & Tan, F. (2024). Enhancing the power of OOD detection via sample-aware model selection. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 17148–17157.
2. Yoav Benjamini and Yosef Hochberg. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal statistical society: series B (Methodological), 57(1):289–300, 1995. 
3. Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
4. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
5. Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. 2011.
6. Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015.
7. Pingmei Xu, Krista A Ehinger, Yinda Zhang, Adam Finkelstein, Sanjeev R Kulkarni, and Jianxiong Xiao. Turkergaze: Crowdsourcing saliency with webcam based eye tracking. arXiv preprint arXiv:1504.06755, 2015.
8. Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3606–3613, 2014.
9. Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. IEEE transactions on pattern analysis and machine intelligence, 40(6):1452–1464, 2017.
10. Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
11. Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The inaturalist species classification and detection dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8769–8778, 2018.
12. Jianxiong Xiao, J Hays, KA Ehinger, A Oliva, and A Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2010.

# Contact

- [Esra Genç](mailto:esra.genc@metu.edu.tr)
- [Emirhan Yılmaz Güney](mailto:email@metu.edu.tr)
