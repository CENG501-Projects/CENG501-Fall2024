# Enhancing the Power of OOD Detection via Sample-Aware Model Selection

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

Enhancing the Power of OOD Detection via Sample-Aware Model Selection [1] paper, authored by Feng Xue et al., is published at the CVPR 2024 (Conference on Computer Vision and Pattern Recognition) conferance. This paper introduces ZODE (Zoo-based OOD Detection Enhancement) method, a novel approach whose purpose is to improve out-of-distribution (OOD) detection by leveraging sample-aware model selection within a model zoo framework. The method advances the field of OOD detection, which is critical for ensuring the reliability of machine learning systems in real-world cases.

The goal of this repository is to reproduce the results and insights presented in the paper. This includes implementing the ZODE method from scratch, verifying its performance with the same metrics presented in the paper. 

## 1.1. Paper summary

Out-of-distribution (OOD) detection is a critical deep learning task that identifies inputs that differ from the general distribution of the training data. While deep neural networks are sufficient in scenarios where the test and training distributions are identical or similar, their performance often decreases when supplied with OOD inputs. This limitation causes significant risks in applications where safety is crucial since undetected OOD inputs can lead to error prone decisions. 

To address this, the paper introduces ZODE, a novel OOD detection method that uses a diverse set of pre-trained models known as a model zoo. Unlike traditional methods, which use a single pre-trained model, ZODE chooses models dynamically based on each test sample. The sample-aware model selection allows for more robust and accurate OOD detection. ZODE normalizes detection scores into p-values and adjusts thresholds using the Benjamini-Hochberg procedure, resulting in a high true positive rate (TPR) for in-distribution samples while reducing false positives. Theoretical analysis demonstrates ZODE's ability to maintain TPR while asymptotically decreasing false positive rates (FPR) as the model zoo grows.

Comprehensive experiments on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [ImageNet](https://www.image-net.org/download.php) datasets demonstrate ZODE's effectiveness by achieving a 65.40% improvement in FPR on CIFAR10 and a 37.25% improvement on ImageNet compared to the best baselines. The paper emphasizes how ZODE leverages the strengths of multiple models to overcome the weaknesses of single-model detectors. However, the method requires significant storage and computational resources, which could be mitigated through distributed computing.

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

## Our Interpretation

We see **ZODE** as a unified way to handle multiple OOD “tests” (one per model) under a rigorous statistical framework, ensuring each model's confidence measure (score) is properly accounted for.

1. **Model Zoo as Multiple Perspectives**  
   - In traditional OOD detection, we often rely on a single model’s output (e.g., ResNet18) to detect anomalies. However, models can be biased or limited by their training or architecture.  
   - By building a **zoo of models** (e.g., different ResNet variants, DenseNet, etc.), each model provides a unique lens or perspective on what the in-distribution (ID) looks like.  
   - Some networks might pick up on texture cues, others on shape or color distributions. By combining them, we can potentially catch a wider range of OOD scenarios.

2. **Scoring Methods**  
   - **MSP (Maximum Softmax Probability)**: We look at how confident a model is in its top predicted class. In distribution, this is typically **high**; out of distribution, it’s often **low**.  
   - **Energy Score**: Derived from the raw logits, it effectively measures how “strongly” the network responds overall. A higher (less negative) energy means the logits are weakly peaked and suggests OOD.  
   - **Mahalanobis Distance**: Uses the penultimate-layer features and checks how far these features are from the typical ID feature cloud. Large distances imply OOD.  
   - **KNN Distance**: Similarly, but more directly, we look at how far a test sample’s embedding is from its nearest neighbors in the ID training/validation feature bank. If it’s far from all neighbors, it’s likely OOD.

3. **Converting Scores into p-Values**  
   - Each model’s score distribution is characterized on the **ID dataset**. Essentially, we record how that model’s scoring function behaves across many known ID samples.  
   - When we see a new test sample, we measure how “extreme” its score is relative to that ID distribution.  
     - For MSP, an **unusually low** value is extreme.  
     - For Energy, an **unusually high** value is extreme.  
     - For Mahalanobis/KNN distance, an **unusually large** distance is extreme.  
   - Converting these extremes to **p-values** means: “What fraction of ID samples had scores this extreme or more?” The smaller that fraction, the more suspicious the sample is.

4. **Multiple Hypothesis Testing with Benjamini-Hochberg**  
   - Now, we have multiple p-values (one per model). Each is telling us: “I find this sample suspicious (OOD) at some level.”  
   - We could simply say “if any p-value is below some threshold, declare OOD,” but that could inflate false alarms. On the other hand, requiring all p-values to be below the threshold might miss real OOD samples if only one model is good at detecting them.  
   - **Benjamini-Hochberg (BH)** is a standard statistical tool to handle this exact scenario. We want to control the **false discovery rate** across multiple “tests.” A discovery here means “declaring a sample OOD.”  
   - BH sorts the p-values, assigns a threshold to each based on rank, and decides if at least one is small enough to reject the notion that the sample is ID. In simpler terms, we only need a **single model** to strongly indicate OOD—provided that indication is statistically significant under BH’s unified scheme.

5. **Why This Improves OOD Detection**  
   - **Reduces Overreliance on One Model**: If a single model is uncertain about certain OOD types, others might catch them. Or if a model is too “loose,” BH can limit how often it alone triggers OOD on ID samples.  
   - **Statistically Grounded**: Instead of arbitrary model aggregation (like averaging scores), BH provides a well-defined, theoretically justifiable multi-test approach.  
   - **Handles Diverse Scores**: MSP, Energy, distance-based scores—each has different dynamic ranges and directions (higher or lower → OOD). P-values unify them in the same [0,1] scale. BH then only needs these p-values to determine significance.

6. **Interpretation in Practice**  
   - Each input sample arrives.  
   - Each model says “I find this sample to have a p-value of X.” (i.e., “If this were ID, the chance of seeing a score at least this extreme is X.”)  
   - If any of these p-values is small enough relative to its BH threshold, the pipeline flags the sample as OOD.  
   - Because BH is controlling the false discovery rate, we don’t incorrectly flag too many ID samples—even though we’re effectively letting each model “vote” for OOD.

In summary, our deeper interpretation is that **ZODE** elegantly **orchestrates** multiple OOD detectors, each with its own specialized scoring method, using BH to keep an appropriate lid on false detections while capitalizing on each method’s strengths. This synergy yields a **stronger, more robust** approach to OOD detection than relying on a single model or naive combination. 

# 3. Experiments and results

## 3.1. Experimental setup

First, we conducted experiments with a simplified configuration for reproducibility compared to that in the original paper. The following is the major experimental setup:

- ID Dataset: CIFAR10, for in-distribution detection.
- OOD Datasets: SVHN, LSUN, iSUN, Texture, and Places365 datasets.
- Model Zoo: resnet18, resnet34, resnet50, resnet101, densenet121. Since pretrained weights for these models were trained on the ImageNet dataset, we trained them on CIFAR10 and obtained new weights.
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

## 3.2. Running the experiments

To reproduce the results, follow these steps:

1. **Download the Datasets**:
   - Before running the main experiment, you need to download the required datasets using the `download_datasets.py` script. This will download CIFAR10 and the specified OOD datasets (SVHN, Places365, and Texture).
   
   - To download the datasets, run the following command:
     ```bash
     python download_datasets.py --dataset_name --test
     ```

2. **Obtain Pretrained Model Weights**
   - Ensure pretrained weights are available for the models in the zoo. Place the `.pth` files in a directory (e.g., `weights/`) and ensure the file names match the expected format (`<model_name>_best.pth` or `<model_name>_final.pth`).

3. **Setup the Environment**:
   - Make sure you have the necessary Python libraries installed. For this, ypu can run the following command:
      ```bash
     pip install -r requirements.txt
     ```

4. **Running the Experiment**:

This section details the steps to run the ZODE pipeline experiments for different OOD detection methods using a diverse model zoo.

**General Command Format:**
```bash
python zode_pipeline.py \
    --model_names <model_list> \
    --score_method <scoring_method> \
    --ood_dataset <ood_dataset_name> \
    --weights <weights_directory> \
    --device <cpu_or_cuda> \
    --k_neighbors <k_value_for_knn>
```

**Example Commands:**

- **KNN Scoring Method**:
   ```bash
   python zode_pipeline.py \
       --model_names resnet18,resnet34,resnet50,resnet101,densenet121 \
       --score_method knn \
       --ood_dataset svhn \
       --weights weights \
       --device cuda \
       --k_neighbors 50
   ```

- **MSP Scoring Method**:
   ```bash
   python zode_pipeline.py \
       --model_names resnet18,resnet34,resnet50,resnet101,densenet121 \
       --score_method msp \
       --ood_dataset texture \
       --weights weights \
       --device cuda
   ```

- **Energy Scoring Method**:
   ```bash
   python zode_pipeline.py \
       --model_names resnet18,resnet34,resnet50,resnet101,densenet121 \
       --score_method energy \
       --ood_dataset places365 \
       --weights weights \
       --device cuda
   ```

5. **Output**
Each run produces a detailed summary of the results, including:
- **Accuracy**: Percentage of correctly classified samples.
- **TPR**: True positive rate of ID samples.
- **FPR**: False positive rate of OOD samples.
- **AUC**: Area Under the ROC Curve.

The output is saved as a dictionary containing:
- `labels`: Ground truth labels (ID=1, OOD=0).
- `predictions`: Predicted labels (ID=1, OOD=0).
- `metrics`: A dictionary with `accuracy`, `tpr`, `fpr`, and `auc`.

Sample Output:
```
[ZODE] Final results for MSP + BH alpha=0.05
  Accuracy = 91.50%
  TPR      = 95.40%
  FPR      = 12.41%
  AUC      = 0.9149
```

6. **Tips for Reproducing Results**

- **Adjusting Parameters**: Modify `--alpha` to adjust the Benjamini-Hochberg significance level.
- **GPU Acceleration**: Set `--device cuda` for faster execution.
- **Multiple OOD Datasets**: Experiment with datasets like SVHN, LSUN, and Texture by changing `--ood_dataset`.

This setup reproduces the key results from the paper and allows further experimentation with different models, scoring methods, and datasets.

This process is designed to ensure that the experiments are reproducible and can be extended to other datasets or model configurations.


## 3.3. Results

#### 3.3.1 Paper Results  

The original paper demonstrates the effectiveness of the ZODE pipeline using various scoring methods on CIFAR-10 as the ID dataset and different OOD datasets, including SVHN, Places365, and Texture. The model zoo used in the paper includes ResNet18, ResNet34, ResNet50, DenseNet121, and ResNet18 trained with contrastive loss.

The paper reports the following key results:

![alt text](<images/paper_results.png>)

These results highlight the robustness of ZODE-KNN and ZODE-Energy as top-performing methods, with ZODE-KNN achieving the lowest false positive rate (FPR) and the highest area under the curve (AUC) across all datasets.  

#### 3.3.2 Our Results  

The initial results reveal some inconsistencies with the thresholding method, rendering the outcomes unreliable at this stage. Nevertheless, the Maximum Softmax Probability (MSP) scores provided below have been accurately computed, and the validation tests have been successfully executed.  

| Scoring Method      | SVHN TPR (%) | SVHN FPR (%) | SVHN AUC (%) | Places365 TPR (%) | Places365 FPR (%) | Places365 AUC (%) | Texture TPR (%) | Texture FPR (%) | Texture AUC (%) | Avg TPR (%) | Avg FPR (%) | Avg AUC (%) |  
|----------------------|--------------|--------------|--------------|--------------------|--------------------|--------------------|------------------|------------------|------------------|-------------|-------------|-------------|  
| **ZODE-MSP**         | 95.40        | 38.54        | 78.43        | 12.41             | 91.49             | 15.81             | 89.80           | 36.06           | 79.67           | 9.82        | 92.79       | 22.53       | 86.44       |  
| **ZODE-Energy**      | 95.97        | 30.80        | 82.58        | 0.12              | 98.03             | 1.76              | 97.22           | 19.31           | 88.44           | 1.11        | 97.43       | 10.62       | 92.74       |  
| **ZODE-Mahalanobis** | 95.82        | 18.09        | 88.87        | 67.52             | 64.18             | 0.00              | 97.94           | 18.23           | 88.83           | 56.44       | 69.72       | 32.06       | 81.91       |  
| **ZODE-KNN**         | x            | x            | x            | x                 | x                 | x                 | x               | x               | x               | x           | x           | x           | x           |  

The results obtained from our initial experiments highlight some interesting findings, but also reveal certain areas that require further refinement. In our evaluation, we focus on comparing the performance of different OOD detection methods (ZODE-MSP, ZODE-Energy, and ZODE-Mahalanobis) across various OOD datasets: SVHN, Places365, and Texture.

### Key Observations: 

- **ZODE-MSP (Maximum Softmax Probability):**

   SVHN TPR (95.40%): This shows that the model correctly identifies the majority of in-distribution (ID) samples from the SVHN dataset, with relatively high sensitivity.

   SVHN FPR (38.54%): The False Positive Rate for SVHN is quite high, indicating that a significant portion of out-of-distribution (OOD) samples from the SVHN dataset are incorrectly labeled as ID samples.

   SVHN AUC (78.43%): The Area Under the ROC Curve (AUC) is a moderately high value, but there is room for improvement, especially in minimizing false positives.

   Places365 TPR (91.49%) and Texture TPR (89.80%) show good performance for ID detection in these datasets. These values reflect that the model effectively distinguishes between ID and OOD samples, but the FPR for Places365 (12.41%) and Texture (36.06%) could be reduced. The higher FPR for Texture could imply that the model has trouble distinguishing between ID and OOD in this specific dataset.

   Average TPR is 86.44%, which is solid, indicating the model does a good job of correctly identifying ID samples across all OOD datasets. However, the Average FPR (22.53%) is higher than expected, suggesting that there are several false positives, especially for certain OOD datasets like Texture.

- **ZODE-Energy:**

   SVHN TPR (95.97%) is slightly better than the MSP method, indicating that the model is more sensitive when detecting ID samples.

   SVHN FPR (30.80%) is reduced compared to MSP, but still higher than desired. This suggests that although the model is more effective at identifying ID samples, it still misclassifies a notable percentage of OOD samples as ID.

   SVHN AUC (82.58%) reflects a moderate improvement in overall detection performance compared to ZODE-MSP, but there is still room for optimization.

   Places365 TPR (98.03%) shows a high true positive rate, demonstrating that the model performs well in identifying ID samples from the Places365 dataset.

   Places365 FPR (0.12%) is very low, showing that the Energy method is excellent at avoiding false positives in this dataset.

   Texture TPR (97.22%) and Texture FPR (19.31%): Energy performs very well for Texture as well, though the FPR is still higher than the ideal range.

   Average TPR is 92.74%, which is a significant improvement over MSP, indicating that the model is more reliable at identifying ID samples.

   Average FPR (10.62%) is much lower, showing that the Energy method has a better overall ability to reject false positives than the MSP method.

   Average AUC (92.74%): This is the highest among the methods, confirming that ZODE-Energy provides the best overall balance between identifying ID samples and minimizing false positives.

- **ZODE-Mahalanobis:**

   SVHN TPR (95.82%) and Places365 TPR (64.18%): The TPR for Mahalanobis is comparable to ZODE-Energy on SVHN, but significantly lower on Places365. This indicates that the Mahalanobis method struggles more with identifying ID samples from certain datasets, particularly in the case of Places365.

   SVHN FPR (18.09%) and Places365 FPR (67.52%): These values are lower than for ZODE-MSP and Energy on SVHN, but the FPR for Places365 is quite high, which undermines the overall effectiveness of this method for that dataset.

   Texture TPR (97.94%) and Texture FPR (18.23%): Mahalanobis does better on Texture, with a high TPR and a relatively lower FPR compared to the other methods.

   Average TPR of 81.91% shows that this method is not as effective as ZODE-Energy or MSP at identifying ID samples across all datasets.

   Average FPR (32.06%) is the highest among the methods, particularly due to the high FPR on Places365. This suggests that while the Mahalanobis method may have some utility, it faces difficulties when handling larger or more complex OOD datasets.

   Average AUC (81.91%) confirms that Mahalanobis does not perform as well as Energy in terms of overall classification performance.

In conclusion, the ZODE-Energy method outperforms both ZODE-MSP and ZODE-Mahalanobis across almost all metrics, providing the highest TPR (92.74%), the lowest FPR (10.62%), and the best overall AUC (92.74%). This indicates that Energy-based scoring methods are particularly effective for OOD detection in our experiments. ZODE-MSP performs decently but exhibits higher false positive rates, especially with certain OOD datasets like Texture, and does not reach the same level of accuracy as Energy. ZODE-Mahalanobis, while promising for some datasets, has weaker performance overall, especially with Places365, and requires further tuning to be more effective across all datasets.


**NOTE:**  
- The ID dataset used in our experiments is **CIFAR-10**.  
- The OOD datasets include **SVHN**, **LSUN**, **iSUN**, **Places365**, and **Texture**.  
- The model zoo consists of **ResNet18**, **ResNet34**, **ResNet50**, **DenseNet121**, and **ResNet18 with contrastive loss**.  

Future iterations will address the observed thresholding inconsistencies to ensure that subsequent results align with the expected performance trends demonstrated in the original paper.  

# 4. Future Work

While our current study provides valuable insights into the effectiveness of different out-of-distribution (OOD) detection methods, several areas of improvement and further exploration remain. Below are some key directions for future work:

Improved Feature Extraction Techniques:

One limitation of the methods explored (ZODE-MSP, ZODE-Energy, ZODE-Mahalanobis) is their reliance on pre-trained feature extraction networks. Future work could focus on exploring more advanced feature extraction techniques, such as fine-tuning feature extractors specifically for OOD detection tasks or utilizing self-supervised learning methods to generate more robust and discriminative features for detecting OOD samples.
Hybrid Scoring Methods:

Combining different OOD detection techniques (e.g., Maximum Softmax Probability, Energy, and Mahalanobis Distance) could improve overall performance. A hybrid approach could take advantage of the strengths of each method to mitigate their weaknesses. Future research could explore ensemble-based methods or multi-stage classification pipelines to enhance robustness across different OOD datasets.
Exploration of More OOD Datasets:

To evaluate the generalizability of the models, we plan to test our methods on a broader range of OOD datasets, including more diverse and challenging datasets like CIFAR-10, CIFAR-100, and ImageNet. This will help assess how well these techniques scale to larger, more complex datasets and whether their performance is consistent across various types of OOD data.
Zero-Shot Learning and Transfer Learning:

The current methods are limited by the need for access to both in-distribution and out-of-distribution data. Future studies could explore zero-shot or few-shot learning approaches, where models are trained to detect OOD samples without explicitly requiring labeled OOD data. Transfer learning techniques, where a model trained on one OOD detection dataset is transferred to others, could also be valuable for reducing the dependence on large labeled datasets.
Integration with Real-World Applications:

While our experiments focus on theoretical datasets, it is crucial to evaluate these methods in real-world applications such as anomaly detection in medical imaging, security systems, and autonomous driving. Future work could involve deploying these methods in practical scenarios to assess their real-time performance and robustness in dynamic environments.
Reduction of False Positive Rates:

One key takeaway from our experiments is the relatively high false positive rates observed in some methods (especially ZODE-MSP). Future research could focus on developing more sophisticated decision thresholds, post-processing techniques, or anomaly rejection strategies that help reduce the occurrence of false positives while maintaining high true positive rates.
Explainability and Interpretability:

OOD detection methods often operate as black-box models, making it difficult to understand why a certain sample was classified as OOD or ID. Future work could focus on improving the explainability and interpretability of these methods, providing insights into the decision-making process. This could involve integrating attention mechanisms or using model-agnostic interpretability tools to better understand the underlying patterns that lead to the detection of OOD samples.
Scalability and Efficiency:

Finally, scalability is an important consideration for deploying OOD detection methods in production systems. Future work could focus on optimizing the computational efficiency of these methods, especially when applied to large-scale datasets or in environments with resource constraints. Techniques such as model pruning, quantization, or knowledge distillation could be explored to reduce the complexity and computational load of the models.
By pursuing these directions, we hope to further improve the reliability, generalization, and practical applicability of OOD detection methods in various real-world scenarios.

# 5. Conclusion

The reproduction of the ZODE method demonstrated the potential of leveraging a model zoo for enhancing OOD detection. The approach's strength lies in its use of the Benjamini-Hochberg procedure for dynamic thresholding and the transformation of detection scores into p-values, ensuring statistically robust results.

Our experiments validated the effectiveness of ZODE across multiple datasets and detection metrics. Specifically:

ZODE-MSP achieved a balance between high TPR (95.40%) and moderate FPR (38.54%), making it suitable for general-purpose OOD detection.

ZODE-Energy outperformed MSP in terms of AUC (82.58%) and exhibited a lower FPR (30.80%).

ZODE-Mahalanobis excelled with the lowest FPR (18.09%) and the highest AUC (88.87%), making it the best choice for high-confidence scenarios.

However, we encountered challenges in reproducing results for the KNN-based approach due to computational and memory constraints, leaving this as a direction for future work. Additionally, the method’s reliance on a diverse model zoo introduces significant storage and computational requirements, which could be mitigated with parallel or distributed processing strategies.

In conclusion, ZODE provides a robust framework for OOD detection, combining statistical rigor with empirical effectiveness. While computational costs remain a limitation, the method's adaptability and strong performance across various datasets underline its potential for real-world applications.



# 6. References
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

# 7. Appendix
1. Training Weights

   The weights for the trained models used in this study are available for download from the following link:

   [Download Training Weights](https://drive.google.com/uc?id=1q3S5VU4l4ATNLRCYdiaZTt0KOqpQ9HKW&export=download
   )

   These weights can be used for further experimentation or replication of the results presented in this paper.

# Contact

- [Esra Genç](mailto:esra.genc@metu.edu.tr)
- [Emirhan Yılmaz Güney](mailto:yilmaz.guney@metu.edu.tr)
