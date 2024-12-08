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
This methodology is validated empirically through enormous tests among multiple datasets and attack scenarios. The effectiveness of defense is gauged by its ability to maintain high model accuracy while identifying and mitigating backdoor influences.

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
We used a similar method to the original method, but with the simplifications to validate the effectiveness of our implementation of DTINSPECTOR. This will be about measuring the attack success rate(ASR) against the accuracy of the model.


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

## **7.1 Prerequisites**

Before running the code, ensure the following are installed:

Python 3.7+

**Required Libraries:**

```pip install numpy tensorflow matplotlib scikit-learn```

## 7.2 Directory Structure

Since all code is in a single file, there’s no complex directory structure required. However for clarity, here’s how you can organize your project:

project/

main.py          (The complete implementation script)     

README.md        (report)

requirements.txt  (List of required libraries)

## 7.3 Running the Code

**Clone the Repository or Copy the Script:**

Save the code in a file named main.py.

**Execute the Script:**

Run the script using: `python main.py`

## 7.4 Code Overview


### 7.4.1. Dataset Loading and Preprocessing

The CIFAR-10 image dataset consists of 60,000 32 by 32 coloured images of 10 different types and is implemented on the TensorFlow Keras API. The pixel values are scale down to the portion [0, 1] through division to the value of 255. The processed raw training data is divided into both training and validation for back propagation performance check having 80% training and 20% validation data.

```
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()  # Flatten labels
x_train = x_train / 255.0
x_test = x_test / 255.0

# Split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
```

### 7.4.2. Trigger Injection

In order to mimic a backdoor attack the red 3x3 trigger is inserted into the bottom right in a part of the images. The poisoned images are given a specific target label (in general, 0) whereas other images are classified as clean. This results in a poisoned training dataset in which only a few of the samples include the trigger.

```
def add_trigger(images, trigger_size=3, color=(1, 0, 0)):
    images_with_trigger = images.copy()
    for img in images_with_trigger:
        img[-trigger_size:, -trigger_size:, :] = color  # Add trigger to bottom-right corner
    return images_with_trigger

def poison_data(x, y, target_label=0, poison_rate=0.1):
    num_poison = int(len(x) * poison_rate)
    poisoned_images = add_trigger(x[:num_poison])  # Apply trigger
    poisoned_labels = np.full((num_poison,), target_label)
    x_clean, y_clean = x[num_poison:], y[num_poison:]
    x_poisoned = np.concatenate([poisoned_images, x_clean], axis=0)
    y_poisoned = np.concatenate([poisoned_labels, y_clean], axis=0)
    return x_poisoned, y_poisoned
```

### 7.4.3. Model Architecture

To classify the CIFAR-10 images, a shallow Convolutional Neural Network (CNN) is proposed and applied. The architecture includes:

1. Three convolutional layers used in extracting the features and ReLU activation function used.

2. Two max-pooling layers in order to reduce the dimensionality.

3. Finding an optimal number of hidden layers remains relatively easy while determining the number of neurons that should be present in any of the hidden layers is quite challenging Hence, the middle layer could be fully connected and associated with ReLU activation.

4. A dropout layer, that helps with preventing overfitting.

5. A suitable output layer which is a softmax layer for multi class classification.

```
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

### 7.4.4. Training and Evaluation

First, the clean model is trained using the original data to set the benchmark of performance. Then, a poisoned model is trained with the poison, unconsciously feeding the model the poisoned dataset. Both models are assessed for the Benign Accuracy (BA) on clean test data, and the poisoned model is also analyzed for the attack success rate (ASR) on triggered inputs.

```
# Train clean model
clean_model = create_model()
clean_model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_val, y_val), verbose=1)
original_accuracy = clean_model.evaluate(x_test, y_test, verbose=0)[1] * 100
print(f"Original Accuracy (Clean Model on Clean Test Data): {original_accuracy:.2f}%")

# Poisoned training data
x_train_poisoned, y_train_poisoned = poison_data(x_train, y_train)

# Train poisoned model
poisoned_model = create_model()
history = poisoned_model.fit(x_train_poisoned, y_train_poisoned, epochs=20, batch_size=64, validation_data=(x_val, y_val), verbose=1)
benign_accuracy = poisoned_model.evaluate(x_test, y_test, verbose=0)[1] * 100
x_test_triggered = add_trigger(x_test)
attack_success_rate = poisoned_model.evaluate(x_test_triggered, np.zeros_like(y_test), verbose=0)[1] * 100
print(f"Benign Accuracy: {benign_accuracy:.2f}%, ASR: {attack_success_rate:.2f}%")
```

### 7.4.5. Patch Learning and Anomaly Detection

An adversarial 3x3 patch is generated with high confidence samples to manipulate the decision of the model. To make a patch effective it is best to incorporate the confidence reduction and the complexity of the patch. Low-confidence samples are used with patch to compute the transfer ratio, and Median of the Absolute Deviation is used to identify the outliers on the labels.

```
def learn_patch(model, high_conf_data, target_label, lambda_value=0.0001, iterations=200, patch_size=3):
    patch = tf.Variable(np.random.rand(patch_size, patch_size, 3).astype(np.float32), trainable=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            patched_data = high_conf_data.copy()
            for img in patched_data:
                img[-patch_size:, -patch_size:, :] = patch.numpy()
            predictions = model(patched_data, training=False)
            loss = tf.reduce_mean(predictions[:, target_label]) + lambda_value * tf.reduce_sum(tf.abs(patch))
        gradients = tape.gradient(loss, [patch])
        optimizer.apply_gradients(zip(gradients, [patch]))
    return patch.numpy()
```

The figure below shows the original, poisoned and patched images.

<img width="659" alt="image" src="https://github.com/user-attachments/assets/c6590b56-b862-46a6-aaa9-5aa5aa8199fa">

### 7.4.6. Visualizations

TSeveral visualizations are generated to analyze the models and predictions:

1. Training vs. validation loss curves.

2. A table summarizing original accuracy, benign accuracy, and ASR.

3. Histograms of prediction confidences for clean and poisoned samples.

4. Bar charts and boxplots comparing transfer ratios for infected and clean labels.

```
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='x')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
```

## 7.5 Notes for Users

**Hardware Requirements:** It is intended for standard CPU or single gpu. Training time may extend depending on the training hardware you are using.

**Customization:** To change between different poison experiment results, change the poison_rate variable on the poison_data function.

**Patch Learning Parameters:** In the following learn_patch function, the values for patch size, the number of iterations, and optimization hyperparameters can be tuned.

# 8. Results

## 8.1 Evaluation of Backdoor Attack:

The following table shows the performance of the BadNet attack on the CIFAR-10 dataset:

<img width="365" alt="image" src="https://github.com/user-attachments/assets/c0750df3-0fcd-4129-8be3-171a32a1d325">

**Accuracy Comparison:**

The original accuracy and benign accuracy in this experiment are lower than that in the original paper, which suggests that the backdoor attack weakens the model’s robustness. The lower benign accuracy show that the attack affects its capacity to classify normal (benign) samples.

**Attack Success Rate (ASR):**

The BadNet attack’s Attack Success Rate (ASR) in this experiment is **99.70%** while the one obtained in the original paper is only 96.34%; therefore, the misclassification of the poisoned trigger in the test data was more successful in the our experiment.

## 8.2 Analysis of Training and Validation Loss:

**Training Loss (Blue Line):** Training loss is initiated at a higher level and decreasing with epochs on the graph below, it gradually reduces with the increase of epochs. Indeed this shows that the model is learning and is able to recognize patterns in images which is the general aim of every epoch.

**Validation Loss (Orange Line):** The validation loss is also smaller in the beginning but looks like it flattens after the 8th epoch. This might mean that the models generalization to unseen data becomes stable and may suggests where the over-fitting starts to offset increased Model complexity.

Immediately after, the training and validation losses sharply drop, rather low up to the five epochs, implying that the model is quickly learning patterns from the training data. The gradually decelerated rate of the validation loss indicates that although the model is still being trained it ceases to generalize over the training data set very well at a later point of time.

Further analysis of this behavior could be done to see whether the model benefits from early stopping or some other kind of regularization.

<img width="417" alt="image" src="https://github.com/user-attachments/assets/4f7878eb-15b9-413f-8739-2d6fed883274">

## 8.3 Comparison of Transfer Ratio Visualizations for Detecting Infected Labels

These two visualizations demonstrate different transfer ratio patterns for CIFAR-10 under the BadNet attack. Our visualization for transfer ratio shows that for clean labels it is approximately 0.18 and for infected labels it is approximately 0.10. This hints at a clear separation between the ‘clean’ and ‘infected’ labels in the model and yet the visualisation does not allow one to see much variability in between.

On the other hand, the transfer ratios have been illustrated in detail in the paper’s visualization. In clean label transfer ratios, they are close to 1.0 while in infected labels, they are close to 0. This wide range and separation underline a significantly increased differentiation between clean and infected labels. In comparison, the scale in the paper’s figure is different together with the approach to differentiate and compare the results by some other ways. Both maps seek a clear separation, but the nature and degree of this are different.

Our experiment visualization:

<img width="551" alt="image" src="https://github.com/user-attachments/assets/efb1e1f6-9c51-4ddb-ad4c-50e1b6c1adc5">

Original Paper Visualization:

<img width="206" alt="image" src="https://github.com/user-attachments/assets/a95cbfea-c0fa-483d-8510-fa7718e67e4e">

## 8.4 Analysis of Transfer ratios for each label:

This figure shows the transfer ratio of each label in the given dataset whereby on the horizontal axis is shown the labels from 0 to 9 and on the vertical axis is shown the transfer ratio of each label. The findings reported here show that the transfer ratio has risen and stabilized at a level close to 0.20 for all labels.

Consistency of this nature indicates that the model does not differentiate between the labels as far as the transfer ratios are concerned. This could suggest that the model has an adequate strategy of label transfer or it could also be a sign that the impact of the attack affects all labels uniformly hence minimal variation. This pattern is different from the general experience where some labels may have higher transfer ratios, which would suggest that the attack affects these classes to a greater extent.

<img width="477" alt="image" src="https://github.com/user-attachments/assets/1b80b1c3-28a3-4084-81d4-22e7de624031">

## 8.5 Analysis of Trojan Detection Results:

Our implementation results differ significantly than the paper’s outcomes for CIFAR-10 on BadNet. Our visualization presents an anomaly index of zero for both clean and trojaned models, and, therefore, all the points in the chart lay on a horizontal. The threshold indicated by the red dashed line is included in the chart; however, as it will be seen, it is not efficient in segregating the clean and trojaned models because all the values are below this point.

Conversely, on the results in the paper, it is entirely noticeable to distinguish between trojaned and clean models; the former’s anomaly index will be much higher than this threshold value, whereas the latter will be much lower. This different representation helps emphasize the efficacy of the detection approach presented in the paper. 

It is hypothesized that one of the reasons for such deviation can be a miscalculation of anomaly indices. Additionally, differences in the detail of the implementation when compared to the paper’s methodology may also explain the gap.

To resolve such matters, the anomaly indices require computation, all of which being in consonance with the research paper.  Thus, the current results point to a need to improve the methodology of detection of trojaned models. Solving these problems will enable high-quality results production and better illustrate the capacity of our approach to identify trojaned models.

<img width="414" alt="image" src="https://github.com/user-attachments/assets/51e56c89-e781-4fa0-b77d-a3b5c0679a13">

## 8.6 Comparison of Results: Prediction Confidence and BA/ASR vs. Poisoning Rate

In poisoning attacks, our results differ from the findings of the paper. While, in our visualization both BA and poisoning confidence drop and remain relatively low at intermediate poisoning rates (8–10%) before increasing partially, the paper shows BA to be consistently high (~85–90%) and ASR to quickly reach saturation (~95%). Further, our prediction confidence values range between 0.45 and 0.51, which are less, especially for poisoning confidence as compared to the paper where both benign confidence ~0.96 and poisoning confidence ~0.98 values are more stable.

Such variations might be due to variations in model architecture, or in ways of computing corresponding metrics. Adapting the approach used by this paper may help in obtaining the results that align with papers result.

Our Models Visulization:

![image](https://github.com/user-attachments/assets/b6190ca3-6108-40d8-9291-0b350fc7ef11)

Papers visualization:

![image](https://github.com/user-attachments/assets/1b0163bc-688a-46ad-8809-a24db88af797)




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
