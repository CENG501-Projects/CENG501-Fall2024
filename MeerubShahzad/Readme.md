# Inspecting Prediction Confidence for Detecting Black-Box Backdoor Attacks


# 1. Introduction
The paper **"Inspecting Prediction Confidence for Detecting Black box Backdoor Attacks"** from the Thirty Eighth AAAI Conference on Artificial Intelligence was the inspiration for this project. Published by Tong Wang and coworkers, the study proposes DTINSPECTOR, a defence mechanism for backdoor attack detection in deep learning. These attacks leverage on some latent triggers in the training data to change the model’s output while not affecting its performance on clean data.

The goal of this study is to replicate and validate the performance of DTINSPECTOR through controlled testing. For the purpose of this work, due to resource limitations, we have restricted our analysis to CIFAR10 as the dataset and BADNET attack to assess DTINSPECTOR’s potential of detecting trojaned models and infected labels.

## 1.1.	Paper Summary
The paper 'Inspecting Prediction Confidence for detecting Black Box Backdoor Attacks' by Tong Wang et al. (Thirty Eighth AAAI Conference on Artificial Intelligence (AAAI-24)) presents DTINSPECTOR, a novel defense mechanism against black box backdoor attacks on deep learning models. In paper they considered these attacks in which an output of the model is perturbed by injecting a trigger hidden in the training set, which is hard to learn as it can still maintain normal predictive accuracy on inputs.
### 1.1.1.	Key Contributions:
**•	Novel Perspective:** The Paper propose a novel approach that leverages the insight that backdoor attacks typically lead poisoned samples exhibiting higher prediction confidence than clean samples. Theoretical analysis and empirical testing show that this observation is of worth.

**•	DTINSPECTOR Mechanism:** DTINSPECTOR is a defense method which uses a distribution transfer technique that magnifies the difference between normal and poisoned data and is built on top of the prediction confidence evaluation on data samples. This technique works well to detect and identify backdoor triggers independent of the backdoor trigger size or its pattern.

**•	Extensive Evaluation:** Evaluations of DTINSPECTOR against different types of backdoor attacks on distinct datasets are extensively explained in the paper which shows that DTINSPECTOR outperforms other existing defenses like Neural Cleanse (NC), ABS, and others on trained models and trained labels.
### 1.1.2.	Relation to Existing Literature:
The proposed method consists of advanced traditional defenses that either detect small sized triggers or analyze neuron activation patterns, both of which can be beaten by highly complex backdoor strategies. DTINSPECTOR is more adaptive to unseen attacks and more robust to changes in the trigger configurations when compared to previous methods.


### 1.1.3.	Methodology:
DTINSPECTOR first sorts training data by prediction confidence, applies learned patches to high confidence samples, and measures the shift in prediction outputs to detect anomalies. In contrast to previous methods capable of directly analyzing triggers or utilized as adjuncts only by relying on anomalies in model output without factoring in confidence levels, this approach directly analyzes triggers or relies on anomalies only by factoring in confidence levels, i.e., high confidence or low confidence.
# 2. Threat Model
In the paper, a method is designed to address **black box backdoor attacks** on deep learning models where adversaries add malicious triggers to only a small portion of the training data, they do not get access to the model or the training process. When these triggers are activated, only then the model makes wrong predictions. On regular inputs though, it performs normally and its hard to detect. To avoid raising suspicion, attacker only changes few data and labels as input.

As a countermeasure, they proposed an attack detection mechanism called **DTINSPECTOR** that looks for such attacks by analyzing uncharacteristic patterns in the model prediction confidence. Defender is the one who has access to the model and its training data, and use tools like DTINSPECTOR. This tool finds and discards poisoned data effects. This protects the model from this type of attack.

# 3. Key Observations from the Paper
In this research paper, backdoor threats in deep learning models are detected at in depth. An effective black box backdoor attack usually produces unusually high prediction confidence on the poisoned training data. During model inference, this elevated confidence is used to achieve a high Attack Success Rate (ASR), and in that way, it becomes an important indicator of data manipulation.
## 3.1 Theoretical and Empirical Support:
**• Theoretical Analysis:** The paper shows how prediction confidence on poisoned training samples is related to the ASR of poisoned testing inputs. Risk function is used for expressing this relationship. The changes in model behavior due to backdoor attacks are quantified with this risk function.

						![image](https://github.com/user-attachments/assets/28946747-8e9c-4914-9b1b-1af7bacc6b2d)


						![image](https://github.com/user-attachments/assets/6e43e514-a954-4f9e-86b1-85580ad05f37)

**•	Empirical Evidence:** To support the theory, the authors of paper present the empirical evidence showing that with an increase in the poisoning rate, the prediction confidence of poisoned data significantly surpasses that of clean data. This evidence is visually supported by:

						![image](https://github.com/user-attachments/assets/9fa608f4-bfe6-4252-bc28-cbf1de248a80)

## 3.2 Conclusion from Observations
Finally, they conclude that backdoor attacks would not work well without substantial change in prediction confidence while the amount of poisoned data is kept small to negligible. In particular, this conclusion suggests development of defense mechanisms that detect and counteract such attacks by focusing on prediction confidence anomalies.
# 4. The method and our Interpretation
## 4.1. The original method
This paper introduces DTINSPECTOR, a novel defense mechanism to detect black box backdoor attacks by evaluating deep learning model's prediction confidence levels. Uniquely, this method describes an innovative means to identify the effects of backdoor attacks and uses various steps to guarantee that the predictions of the model are not compromised.
### 4.1.1.	Data Segregation and Sampling
**o	Sorting and Selection:** The training data is sorted by the first prediction confidence under each label. The data is then divided to show top K samples with highest confidence and bottom K samples with least confidence.

					![image](https://github.com/user-attachments/assets/056800fe-7c0f-4f80-bb8e-3af693c0e643)

### 4.1.2. Patch Learning and Application
**o	Patch Design:** A specific patch **δ** is designed using only the high confidence samples. This patch is desired to shift the model's predictions from the current label to any other label.

**o	Objective Function:** The patch is optimized through an objective function stated as:  

					![image](https://github.com/user-attachments/assets/d774f4c9-6221-454b-9941-11c6a1dd84ea)

					![image](https://github.com/user-attachments/assets/1a29fca9-fd40-420e-b941-2b7f69700e4a)

These pictures shows examples of the learned patches and how they alter the data samples.
### 4.1.3. Transfer Ratio Computation and Anomaly Detection
**o	Patch Application:** The learned patch is applied to the low confidence samples, and the change in prediction labels is monitored.

**o	Transfer Ratio:** The transfer ratio is calculated as the proportion of low confidence samples whose labels have been changed due to the patch application.

**o	Anomaly Detection:** Anomalies in the transfer ratios are analyzed to identify labels that depicts notably different behaviors, consequently it suggests potential backdoor manipulation.
### 4.1.4.	Effectiveness and Validation
o	This methodology is validated empirically through enormous tests among multiple datasets and attack scenarios. The effectiveness of defense is gauged by its ability to maintain high model accuracy while identifying and mitigating backdoor influences.

This methodology ensures a robust defense against enormous backdoor attacks by focusing on the typical prediction confidences induced by such attacks, providing a significant enhancement in security measures for deep learning models. The detailed and iterative approach of DTINSPECTOR, from data sampling to anomaly detection, offers a comprehensive defense strategy that is adaptable to various attack complexities.
## 4.2.  Our Interpretation of Methodology
### 4.2.1.	Data Segregation and Sampling:
The paper mentions about data categorization by the level of prediction confidence and choosing samples with high and low levels of confidence. However, it was not clear what specific requirements governed these selection decisions. In our implementation, the thresholds have been set based on **median confidence score** from the distribution of the poisoned training data. The high-confidence sample and low-confidence samples for each predictor were defined based on a median value of the confidence of each sample. This method is easy to implement and partitions the data effectively and equitably and incorporates both ends of the prediction spectrum.
### 4.2.2. Patch Learning and Application:
**Patch Design:** The paper discusses designing a patch δ from high-confidence samples but it doesn't specify how the patch dimensions are decided or how it should be applied across different data sizes or formats. In our work, a fixed patch of size **3x3** was placed at the bottom right of the high confidence images. This guarantees uniformity regardless of the size of the data and makes the patch optimizing cycle easier.

**Objective Function:** We quantified the idea of patch learning by minimizing a measure of prediction confidence of the target label and the patch simplicity (the L1 norm of the mask) using **gradient based optimization**. This guarantees the learned patch is well equipped to counter poisoned predictions while at the same time, not too large to cause unnecessary changes.
### 4.2.3. Transfer Ratio Computation and Anomaly Detection:
The concept of the transfer ratio was implemented by applying the learned patch to the low confidence samples and by determining the percentage of samples that receive different label predictions. This value is used to measure anomaly in the behaviour of the label. For anomaly detection, we employed **Median Absolute Deviation (MAD)** to detect labels with low transfer ratios as potential infected labels. This method does not rely on the fact of distribution of labels and delivers a stable measure of anomalies.
### 4.2.4.	Effectiveness and Validation:
In our implemenation, validation was carried out through empirical assessment applying Benign Accuracy (BA), Attack Success Rate (ASR) and transfer ratios for each label. These metrics were calculated on clean and poisoned test datasets to analyze the performance of DTINSPECTOR when it comes to detecting that the model is infected, as well as the labels are infected. We did not employ cross-validation because of limited resources but the approach offers accurate information regarding the defense in this carefully constructed environment.
# 5. Experiments and results
## 5.1 Original Experimental Setup:
**o	Datasets Used:** The original study utilized commonly studied datasets like CIFAR10, GTSRB, ImageNet, and PubFig, chosen for their relevance and challenge for the AI community.

**o	Model Configurations:** Number of models, such as convolutional networks and ResNets, were tailored specific to each dataset.

**o	Backdoor Attacks:** This paper evaluated the defense against six black-box backdoor attacks and demonstrate that DTINSPECTOR behaves robustly.
## 5.2	Adaptations and Changes for My Project:
**o	Dataset Selection:** Due to limited computational resources we couldn't make our model viable for all research problems, we decided to focus on CIFAR10 only. 

**o	Model Simplification:** The original paper employed complex architectures, we used simple Convulutional Neural Network that require lesser computational power but still yield similar results as far as understanding and showcasing defense mechanisms against backdoor attacks are concerned. Here is the architecture of our model:
**Convolutional Layer 1:**

Filters: 32
Kernel Size: (3, 3)
Activation: ReLU
**MaxPooling Layer 1:**

Pool Size: (2, 2)
**Convolutional Layer 2:**

Filters: 64
Kernel Size: (3, 3)
Activation: ReLU
MaxPooling Layer 2:

Pool Size: (2, 2)
**Flatten Layer:**

Translates the feature maps into feature vector where the width of the maps are flattened.
**Dense Layer 1:**

Units: 64
Activation: ReLU
**Dropout Layer:**

Dropout Rate: 0.5
**Dense Layer 2 (Output Layer):**

Units: 10 (correspond to 10 classes in CIFAR10)
Activation: Softmax

**o	Limited Attack Types:** Rather than using six different backdoor attacks, we will focuse on just BADNET. This provide a focused approach to testing and understanding the efficacy of a defense.
## 5.3 Experimental Validation:
o	We will then used a similar method to the original method, but with the simplifications to validate the effectiveness of our implementation of DTINSPECTOR. This will be about measuring the attack success rate(ASR) against the accuracy of the model.


# 6. Challenges and Mitigation in Original Paper:
The paper says that development and implementation of the DTINSPECTOR defense system have faced several critical challenges involving subtlety of backdoor attacks and variability of attack mechanisms. Here we describe some of the challenges, and how we manage these:
## 6.1.	Challenge: Subtlety of Trigger Mechanisms
**o	Description:** Trigger mechanisms in black box backdoor attacks can be very subtle, and being detected is much difficult without sacrificing the performance of general inputs on the model.

**o	Mitigation:** In our implementation we focused on identification of discrepancies in the prediction confidence between clean samples and the poisoned ones. When analyzing the poisoned samples using DTINSPECTOR, what we found was that the confidence scores of these samples were much higher than that of clean samples. Thus, we have taken advantage of this disparity and proposed a 3x3 patch which is localized to the bottom right corner of images to detect poisoned labels without much degradation of the models performance on clean data.
## 6.2. Challenge: Generalization Across Attack Variants
**o Description:** The defense must be strong enough to be able to handle variety of backdoor attacks. These backdoor attacks can include such strategies that model might not have seen in the training phase.

**o Mitigation:** Due to lack of computational resources, the evaluation of DTINSPECTOR was done on a one type of an attack (BADNET) and a dataset (CIFAR10). Despite the restrictions of our work, we have proved the method’s fundamental approach to identify trojaned models and infected labels. This forms the basis for the further analysis on the generalization ability of DTINSPECTOR through testing on other datasets and on different types of attack in the subsequent work.
## 6.3. Challenge: Efficiency and Scalability
**o Description:** In the paper, authors stated that maintaining high detection accuracy while ensuring the defense is scalable and efficient in processing time is important especially for real-time applications.

**o Mitigation:** We have made some changes and simplifications to make the experiment easier. We used a lightweight CNN model and CIFAR10. Although we cannot claim that DTINSPECTOR is optimized to run on a machine that has low GPU and CPU, we showed otherwise by modifying the approaches used in the study based on our machine’s resources. For example:

**Data Segregation:** Employed median confidence thresholds in order to minimize computational costs.
**Patch Learning:** To improve the efficiency, the area of the patch was limited to 3x3.
**Anomaly Detection:** We employed Median Absolute Deviation (MAD) since it is computationally cheaper to perform in order to infect labels. This adaptation establishes how feasible it is to implement DTINSPECTOR in practical settings, particularly within resource-scarce scenarios.


# 7. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

# 8. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 9. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 10. References

Tong Wang, et al., "Inspecting Prediction Confidence for Detecting Black-Box Backdoor Attacks", Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24), 2024.


Gu, T., Dolan-Gavitt, B., & Garg, S. "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain". arXiv preprint arXiv:1708.06733, 2017.


Moosavi-Dezfooli, S. M., et al., "Universal Adversarial Perturbations", Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.


Iglewicz, B., & Hoaglin, D. C., "How to Detect and Handle Outliers", ASQC Quality Press, 1993.


Krizhevsky, A., & Hinton, G., "Learning Multiple Layers of Features from Tiny Images", CIFAR Dataset Documentation, 2009.


Liu, Y., Ma, S., Aafer, Y., et al., "Trojaning Attack on Neural Networks", Proceedings of NDSS, 2018.



# Contact


**Haiqua Meerub**
haiquameerub@gmail.com          

**Umyma Shahzad**
umymashahzad02@gmail.com
