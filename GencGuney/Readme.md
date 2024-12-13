# Enhancing the Power of OOD Detection via Sample-Aware Model Selection

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

Enhancing the Power of OOD Detection via Sample-Aware Model Selection [1] paper, authored by Feng Xue et al., is published at the CVPR 2024 (Conference on Computer Vision and Pattern Recognition) conferance. This paper introduces ZODE (Zoo-based OOD Detection Enhancement) method, a novel approach whose purpose is to improve out-of-distribution (OOD) detection by leveraging sample-aware model selection within a model zoo framework. The method advances the field of OOD detection, which is critical for ensuring the reliability of machine learning systems in real-world cases.

The goal of this repository is to reproduce the results and insights presented in the paper. This includes implementing the ZODE method from scratch, verifying its performance with the same metrics presented in the paper. 

## 1.1. Paper summary

Out-of-distribution (OOD) detection is a critical deep learning task that identifies inputs that differ from the general distribution of the training data. While deep neural networks are sufficient in scenarios where the test and training distributions are identical or similar, their performance often decreases when supplied with OOD inputs. This limitation causes significant risks in applications where safety is crucial since undetected OOD inputs can lead to error prone decisions. 

To address this, the paper introduces ZODE, a novel OOD detection method that uses a diverse set of pre-trained models known as a model zoo. Unlike traditional methods, which use a single pre-trained model, ZODE chooses models dynamically based on each test sample. The sample-aware model selection allows for more robust and accurate OOD detection. ZODE normalizes detection scores into p-values and adjusts thresholds using the Benjamini-Hochberg procedure, resulting in a high true positive rate (TPR) for in-distribution samples while reducing false positives. Theoretical analysis demonstrates ZODE's ability to maintain TPR while asymptotically decreasing false positive rates (FPR) as the model zoo grows.

Comprehensive experiments on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [ImageNet](https://www.image-net.org/download.php) datasets demonstrate ZODE's effectiveness by achieving a 65.40% improvement in FPR on CIFAR10 and a 37.25% improvement on ImageNet compared to the best baselines. The paper emphasizes how ZODE leverages the strengths of multiple models to overcome the weaknesses of single-model detectors.However, the method requires significant storage and computational resources, which could be mitigated through distributed computing.

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

ZODE provides a new way of balancing accuracy and usability in out-of-distribution detection. ZODE standardizes the outputs from different models by converting detection scores into p-values, hence allowing effective integration within the model zoo. More importantly, it leverages the Benjamini-Hochberg procedure to dynamically adjust the detection thresholds, which prevents error accumulation and thus maintains consistent detection performance.

In our reproduction, simplicity and modularity were found to be the strong points of ZODE. The use of the model zoo brings flexibility in the approach, making it appropriate for different datasets and tasks. Regarding the practical implementation of ZODE, there are several challenges to be considered, such as high computational overhead due to model selection for each sample, storage of validation data, and higher memory and processing requirements for large-scale systems; parallel computing and distributed processing can reduce these issues.

The quality and diversity of the model zoo are of prime importance for the success of ZODE. In practice, such a zoo is very resource-expensive to build, which may limit its utility for small teams. However, leaving practical matters aside, ZODE is a theoretically well-grounded and very strong method that improves OOD detection in many important ways.

# 3. Experiments and results

## 3.1. Experimental setup

First, we conducted experiments with a simplified configuration for reproducibility compared to that in the original paper. The following is the major experimental setup:

- ID Dataset: CIFAR10, for in-distribution detection.
- OOD Datasets: SVHN, Places365, and Texture datasets.
- Model Zoo: ResNet18, ResNet34, and ResNet50, all pre-trained on ImageNet.
- Significance Level (α): It has been set to 0.05 to maintain 95% TPR for ID samples.
- Batch Size: 256 for all data loaders.

This setup emphasizes computational efficiency and resource constraints by reducing the number of OOD datasets and models in the zoo. While this is a slight deviation from the original study, it provides a practical baseline for evaluating the performance of ZODE in a controlled environment.

The evaluation was performed by following the same methodology described in the paper:

1. Metrics:

    - TPR for ID samples
    - FPR for OOD samples at 95% TPR.
    - AUC-ROC for overall performance. 

2. Validation Features: features extracted from the CIFAR10 test set for calibrating and adjusting thresholds with ZODE. 
3. Hyperparameter Tuning: The α threshold was adjusted in order to optimally adjust the detection performance while maintaining TPR for ID samples. 

This simplified experimental design therefore allowed us to test the working of ZODE efficiently and to validate its effectiveness under simplified conditions.

## 3.2. Running the code

To reproduce the results, follow these steps:

1. **Download the Datasets**:
   - Before running the main experiment, you need to download the required datasets using the `download_datasets.py` script. This will download CIFAR10 and the specified OOD datasets (SVHN, Places365, and Texture).
   
   - To download the datasets, run the following command:
     ```bash
     python download_datasets.py --data_dir data
     ```

2. **Setup the Environment**:
   - Make sure you have the necessary Python libraries installed, including `torch`, `torchvision`, and `scipy`.

3. **Code Structure**:
   - The experiment is structured into modular components:
     - `data.loaders`: Handles dataset loading and DataLoader creation.
     - `model.zoo`: Manages the pre-trained models in the model zoo.
     - `zode.algorithm`: Implements the ZODE algorithm, including p-value computation and the Benjamini-Hochberg correction.
     - `zode.evaluation`: Provides evaluation metrics such as TPR, FPR, and AUC.

4. **Running the Experiment**:
   - Once the datasets are downloaded, you can run the experiment using:
     ```bash
     python main.py --data_dir data --batch_size 256 --alpha 0.05
     ```
   - This will initialize the model zoo, extract features, and evaluate ZODE on the specified OOD datasets.

5. **Expected Output**:
   - The script will output the results for each OOD dataset, including TPR, FPR, and AUC metrics, which are summarized in the final evaluation.

This process is designed to ensure that the experiments are reproducible and can be extended to other datasets or model configurations.


## 3.3. Results

### 3.3.1. First Results

The initial results reveal some inconsistencies with the thresholding method, rendering the outcomes unreliable at this stage. Nevertheless, the Maximum Softmax Probability (MSP) scores provided below have been accurately computed, and the validation tests have been successfully executed.

NOTE: As mentioned above, the ID dataset used is CIFAR-10 whereas the OOD datasets experimented are SVHN, Places365 and Texture. In addition, the model zoo contains Resnet18, Resnet34, Resnet50, DenseNet121 and Resnet18 with contrastive loss.

### MSP Score Statistics for Datasets

|                Model         |   Dataset   |  Mean   |   Std   |   Min   |   Max   |
|------------------------------|-------------|---------|---------|---------|---------|
| ResNet18                    | SVHN        | 0.2416  | 0.2087  | 0.0086  | 0.9849  |
| ResNet34                    | SVHN        | 0.4132  | 0.2287  | 0.0426  | 0.9983  |
| ResNet50                    | SVHN        | 0.6418  | 0.2507  | 0.0442  | 1.0000  |
| DenseNet121                 | SVHN        | 0.5720  | 0.2736  | 0.0162  | 1.0000  |
| ResNet18_Contrastive        | SVHN        | 0.2416  | 0.2087  | 0.0086  | 0.9849  |
| ResNet18                    | Places365   | 0.4397  | 0.2631  | 0.0226  | 1.0000  |
| ResNet34                    | Places365   | 0.4888  | 0.2685  | 0.0262  | 1.0000  |
| ResNet50                    | Places365   | 0.5028  | 0.2716  | 0.0160  | 1.0000  |
| DenseNet121                 | Places365   | 0.4783  | 0.2660  | 0.0209  | 1.0000  |
| ResNet18_Contrastive        | Places365   | 0.4397  | 0.2631  | 0.0226  | 1.0000  |
| ResNet18                    | Texture     | 0.4166  | 0.2830  | 0.0154  | 1.0000  |
| ResNet34                    | Texture     | 0.4714  | 0.2831  | 0.0161  | 1.0000  |
| ResNet50                    | Texture     | 0.4792  | 0.2896  | 0.0193  | 1.0000  |
| DenseNet121                 | Texture     | 0.4518  | 0.2923  | 0.0178  | 1.0000  |
| ResNet18_Contrastive        | Texture     | 0.4166  | 0.2830  | 0.0154  | 1.0000  |


### ZODE Algorithm Output
|                Model         |    Dataset     |    TP   |    TN   |    FP   |    FN   | Precision |  Recall   |  F1-Score | Threshold  |
|-----------------------------|----------------|---------|---------|---------|---------|-----------|-----------|-----------|------------|
| ResNet18                   | SVHN           |      0  |  10000  |      0  |  26032  | NaN       | 0.000000  | NaN       | 0.008626   |
| ResNet34                   | SVHN           |      0  |  10000  |      0  |  26032  | NaN       | 0.000000  | NaN       | 0.042626   |
| ResNet50                   | SVHN           |  26031  |     13  |   9987  |      1  | 0.722722  | 0.999962  | 0.839033  | 1.000000   |
| DenseNet121                | SVHN           |      0  |  10000  |      0  |  26032  | NaN       | 0.000000  | NaN       | 0.016236   |
| ResNet18_Contrastive       | SVHN           |      0  |  10000  |      0  |  26032  | NaN       | 0.000000  | NaN       | 0.008626   |
| ResNet18                   | Places365      |  35929  |      1  |   9999  |    571  | 0.782290  | 0.984356  | 0.871767  | 0.993372   |
| ResNet34                   | Places365      |      0  |  10000  |      0  |  36500  | NaN       | 0.000000  | NaN       | 0.026214   |
| ResNet50                   | Places365      |      0  |  10000  |      0  |  36500  | NaN       | 0.000000  | NaN       | 0.016026   |
| DenseNet121                | Places365      |      0  |  10000  |      0  |  36500  | NaN       | 0.000000  | NaN       | 0.020905   |
| ResNet18_Contrastive       | Places365      |  35929  |      1  |   9999  |    571  | 0.782290  | 0.984356  | 0.871767  | 0.993372   |
| ResNet18                   | Texture        |   5487  |      1  |   9999  |    153  | 0.354320  | 0.972872  | 0.519455  | 0.993372   |
| ResNet34                   | Texture        |      0  |  10000  |      0  |   5640  | NaN       | 0.000000  | NaN       | 0.016145   |
| ResNet50                   | Texture        |      0  |  10000  |      0  |   5640  | NaN       | 0.000000  | NaN       | 0.019346   |
| DenseNet121                | Texture        |      0  |  10000  |      0  |   5640  | NaN       | 0.000000  | NaN       | 0.017756   |
| ResNet18_Contrastive       | Texture        |   5487  |      1  |   9999  |    153  | 0.354320  | 0.972872  | 0.519455  | 0.993372   |


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
- [Emirhan Yılmaz Güney](mailto:yilmaz.guney@metu.edu.tr)
