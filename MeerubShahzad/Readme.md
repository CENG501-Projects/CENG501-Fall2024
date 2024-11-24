# Inspecting Prediction Confidence for Detecting Black-Box Backdoor Attacks



# 1. Introduction
This project is based on the findings from the paper "Inspecting Prediction Confidence for Detecting Black-Box Backdoor Attacks," presented at the Thirty Eighth AAAI Conference on Artificial Intelligence. The study by Tong Wang and colleagues introduces DTINSPECTOR, a defense strategy against the severe threat of backdoor attacks in deep learning models. These attacks manipulate model predictions by secretly altering training data, it is a vulnerability that DTINSPECTOR counters by examining prediction confidence anomalies.
Replicating and verifying the effectiveness of DTINSPECTOR under controlled experiments is our goal. We aim to reproduce or give evidence to the original results and probe resistance of Defense to new data and attack setups to empirically evaluate the robustness of defense and practical utility in diverse applications and attack scenarios, thereby testing its robustness and practical utility in broader applications.
## 1.1.	Paper Summary
The paper "Inspecting Prediction Confidence for Detecting Black Box Backdoor Attacks" by Tong Wang et al., published at the Thirty Eighth AAAI Conference on Artificial Intelligence (AAAI-24), introduces DTINSPECTOR, a novel defense mechanism against black box backdoor attacks on deep learning models. These attacks usually manipulate an output of the model by inserting hidden triggers in the training data, which are difficult to detect due to their potential to maintain normal model accuracy on the inputs.
### 1.1.1.	Key Contributions:
**•	Novel Perspective:** This paper proposes a unique approach based on the observation that backdoor attacks usually result in higher prediction confidence for poisoned samples as compared to clean samples. This observation has it’s worth through both theoretical analysis and empirical testing.

**•	DTINSPECTOR Mechanism:** The defense method, DTINSPECTOR, evaluates the prediction confidence in the data samples, it utilizes a distribution transfer technique that magnifies differences between normal and poisoned data. This technique effectively identifies the impacts of backdoor triggers without relying on the size or pattern of the trigger.

**•	Extensive Evaluation:** The paper explains extensive evaluations of DTINSPECTOR against several types of backdoor attacks in different datasets, it demonstrates superior performance in identifying trojaned models and infected labels compared to the existing defenses like Neural Cleanse (NC), ABS, and others.
### 1.1.2.	Relation to Existing Literature:
The method proposes advanced traditional defenses that generally detect small sized triggers or analyze neuron activation patterns, which can fail against complex backdoor strategies. Notably, DTINSPECTOR’s ability to adapt to unseen attacks and its robustness against varying trigger configurations stand out as significant improvements over previous methods.


### 1.1.3.	Methodology:
DTINSPECTOR sorts training data by prediction confidence, it applies learned patches to high confidence samples and measures the shift in prediction outputs to detect anomalies. This approach contrasts with earlier methods that can directly analyze triggers or rely solely on anomalies in model output without considering confidence levels.
# 2. Threat Model
The paper addresses the issue of black-box backdoor attacks on deep learning models, where adversaries secretly inject malicious triggers into a small portion of the training data without accessing the model or its training process. These triggers cause the model to make wrong predictions only when activated. Otherwise it performs normally on regular inputs, making the attack hard to detect. The attacker only changes few input data and labels to avoid raising the suspicion.
To counter this, the paper proposes DTINSPECTOR, a defense mechanism that identifies these attacks by analyzing unusual patterns in the model’s prediction confidence. Defender is the one who has full access to the model and its training data and use tools like DTINSPECTOR. This tool identifies and eliminates the effects of poisoned data. It helps in keeping the model safe from such kind of attacks.
# 3. Key Observations from the Paper
The research paper provides a deep insights into detecting backdoor threats in deep learning models. It observes that an effective black-box backdoor attack typically results in unusually high prediction confidence in the poisoned training data. This elevated confidence is utilized to achieve a high Attack Success Rate (ASR) during model inference, making it a critical indicator of data manipulation.
## 3.1 Theoretical and Empirical Support:
**• Theoretical Analysis:** The paper elaborates on the relationship between prediction confidence on poisoned training samples and the ASR on poisoned testing inputs. For expressing this relationship, risk function is used. This risk function quantify the changes in model behavior due to backdoor attacks.

![image](https://github.com/user-attachments/assets/28946747-8e9c-4914-9b1b-1af7bacc6b2d)


![image](https://github.com/user-attachments/assets/6e43e514-a954-4f9e-86b1-85580ad05f37)

**•	Empirical Evidence:** To support the theory, the authors of paper present the empirical evidence showing that with an increase in the poisoning rate, the prediction confidence of poisoned data significantly surpasses that of clean data. This evidence is visually supported by:

![image](https://github.com/user-attachments/assets/9fa608f4-bfe6-4252-bc28-cbf1de248a80)

## 3.2 Conclusion from Observations
The paper concludes that backdoor attacks are less likely to be effective without causing noticeable changes in prediction confidence, especially when the quantity of poisoned data is kept minimal. This conclusion is important for development of defense mechanisms that can identify and neutralize such attacks by focusing on anomalies in prediction confidence.
# 4. The method and our Interpretation
## 4.1. The original method
This paper introduces DTINSPECTOR, a novel defense mechanism designed to detect  black box backdoor attacks through the analysis of prediction confidence levels in deep learning models. This method proposes an innovative approach to discern the manipulations caused by backdoor attacks and employs a number of steps to ensure the integrity of the model predictions.
### 4.1.1.	Data Segregation and Sampling
**o	Sorting and Selection:** First of all, the training data is sorted by the prediction confidence for each label. The data is then divided to select the top K samples and displays the highest confidence and the bottom K samples with the lowest confidence.

![image](https://github.com/user-attachments/assets/056800fe-7c0f-4f80-bb8e-3af693c0e643)

### 4.1.2. Patch Learning and Application
**o	Patch Design:** A specific patch δ is designed using only the high confidence samples. This patch is desired to shift the model's predictions from the current label to any other label.

**o	Objective Function:** The patch is optimized through an objective function stated as: 

![image](https://github.com/user-attachments/assets/d774f4c9-6221-454b-9941-11c6a1dd84ea)

![image](https://github.com/user-attachments/assets/1a29fca9-fd40-420e-b941-2b7f69700e4a)

These pictures shows examples of the learned patches and how they alter the data samples.
### 4.1.3. Transfer Ratio Computation and Anomaly Detection
**o	Patch Application:** The learned patch is applied to the low confidence samples, and the change in prediction labels is monitored.

**o	Transfer Ratio:** The transfer ratio is calculated as the proportion of low confidence samples whose labels have been changed due to the patch application.

**o	Anomaly Detection:** Anomalies in the transfer ratios are analyzed to identify labels that depicts notably different behaviors, consequently it suggests potential backdoor manipulation.
### 4.1.4.	Effectiveness and Validation
This methodology is validated empirically through enormous tests among multiple datasets and attack scenarios. The effectiveness of defense is gauged by its ability to maintain high model accuracy while identifying and mitigating backdoor influences.

This methodology ensures a robust defense against enormous backdoor attacks by focusing on the atypical prediction confidences induced by such attacks, providing a significant enhancement in security measures for deep learning models. The detailed and iterative approach of DTINSPECTOR, from data sampling to anomaly detection, offers a comprehensive defense strategy that is adaptable to various attack complexities.
## 4.2.  Our Interpretation of Methodology
### 4.2.1.	Data Segregation and Sampling:
The paper mentions sorting data by prediction confidence and selecting high and low-confidence samples. But the exact criteria for these selections were not detailed. In our interpretation, we decided to determine the thresholds for high and low confidence. These threshold values will be based on standard deviation metrics from the overall dataset confidence distribution. It will select the most extreme cases that will reveal signs of potential tampering.
### 4.2.2. Patch Learning and Application:
**Patch Design:** The paper discusses designing a patch δ from high-confidence samples but it doesn't specify how the patch dimensions are decided or how it should be applied across different data sizes or formats. We planned to do this by standardizing patch sizes relative to input dimensions and will use the convolutional techniques to ensure comprehensive coverage.

**Objective Function:** The optimization function for patch effectiveness was somewhat abstract. We are planning to operationalize it by using gradient ascent on prediction deviation from actual labels, ensuring the patches not only alter model outputs but do so in a way that would be noticeable without direct comparison to the original output.
### 4.2.3. Transfer Ratio Computation and Anomaly Detection:
The concept of transfer ratio was clear, but its application in a varied dataset environment was not. We decided on using a dynamic normalization process to account for variability in label distribution across datasets. We will ensure the transfer ratio is reliable metric regardless of underlying data skew.
### 4.2.4.	Effectiveness and Validation:
In the paper, validation approach focus on empirical tests without much mention of cross-validation or other statistical validation techniques. Our approach will inlcude a cross-validation framework to robustly assess the DTINSPECTOR's performance across different data partitions and enhancing the reliability of the validation results.
# 5. Experiments and results
## 5.1 Original Experimental Setup:
**o	Datasets Used:** The original study utilized commonly studied datasets like CIFAR10, GTSRB, ImageNet, and PubFig, chosen for their relevance and challenge for the AI community.

**o	Model Configurations:** Number of models, such as convolutional networks and ResNets, were tailored  specific to each dataset.

**o	Backdoor Attacks:** This paper evaluated the defense against six black box backdoor attacks, it showcases the robustness of DTINSPECTOR.
## 5.2	Adaptations and Changes for My Project:
**o	Dataset Selection:** Given the computational constraints, I will focuse on CIFAR10 and GTSRB only. These datasets are smaller and more manageable but they still provide a robust testing ground for backdoor defense mechanisms.

**o	Model Simplification:** Since the original paper can use complex architectures, I will use simple versions of neural networks that require less computational power, they are equally effective for understanding and demonstrating defense mechanisms against backdoor attacks.

**o	Limited Attack Types:** Instead of six different backdoor attacks, I  will concentrate on two only. The BADNET and TROJANNN. These attacks are well documented and more easy to replicate for an undergraduate project, they allow a focused approach on testing and understanding the effectiveness of defense.
## 5.3 Experimental Validation:
To validate the effectiveness of my implementation of DTINSPECTOR, I will follow a similar method to the original method but with the simplifications. This involve measuring the attack success rate (ASR) and comparing it to the accuracy of the model.



# 6. Challenges and Mitigation in Original Paper:
According to the paper, the development and implementation of the DTINSPECTOR defense system have encountered several critical challenges, primarily associated with the subtlety of backdoor attacks and the variability of attack mechanisms. Following are some of the challenges and the strategies used to mitigate them:
## 6.1.	Challenge: Subtlety of Trigger Mechanisms
**o	Description:** The subtlety of trigger mechanisms in black box backdoor attacks makes detection much difficult without majorly affecting the overall performance of the model on the inputs.

**o	Mitigation:** DTINSPECTOR addresses this by utilizing a novel approach that focuses on prediction confidence disparities. The system explains the observation that backdoored models exhibit abnormally high prediction confidence in poisoned samples, it allows effective identification without compromising the performance on samples.
## 6.2. Challenge: Generalization Across Attack Variants
**o Description:** The defense must be strong enough to be able to handle variety of backdoor attacks. These backdoor attacks can include such strategies that model might not have seen in the training phase.

**o Mitigation:** The researchers have tested the proposed defense method against multiple types of backdoor attacks. They have tested it on multiple datasets. They ensured that it is effective across different backdoor attacks. This testing included challenges against advanced attacks that use different trigger mechanisms and target multiple labels.
## 6.3. Challenge: Efficiency and Scalability
**o Description:** In the paper, authors stated that maintaining high detection accuracy while ensuring the defense is scalable and efficient in processing time is important especially for real-time applications.

**o Mitigation:** According to paper, the proposed DTINSPECTOR uses optimized algorithms to balance the computational efficiency. It is implementation such as to minimize overhead and ensure scalability.



# 7. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

# 8. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 9. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 10. References
**Primary Research Paper:**

Tong Wang, et al., "Inspecting Prediction Confidence for Detecting Black-Box Backdoor Attacks", Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24), 2024.

**Backdoor Attacks and Defenses Overview:**

Gu, T., Dolan-Gavitt, B., & Garg, S. "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain". arXiv preprint arXiv:1708.06733, 2017.


**Data Segregation by Confidence:**

Moosavi-Dezfooli, S. M., et al., "Universal Adversarial Perturbations", Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.


**Detection with Anomaly-Based Metrics:**

Iglewicz, B., & Hoaglin, D. C., "How to Detect and Handle Outliers", ASQC Quality Press, 1993.


**Simplified Datasets and Models for Educational Use:**

Krizhevsky, A., & Hinton, G., "Learning Multiple Layers of Features from Tiny Images", CIFAR Dataset Documentation, 2009.

**TrojanNN Attack Methodology:**

Liu, Y., Ma, S., Aafer, Y., et al., "Trojaning Attack on Neural Networks", Proceedings of NDSS, 2018.



# Contact


**Haiqua Meerub**             
haiquameerub@gmail.com          

**Umyma Shahzad**
umymashahzad02@gmail.com
