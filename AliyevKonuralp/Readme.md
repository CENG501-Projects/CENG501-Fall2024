# Active Domain Adaptation with False Negative Prediction for Object Detection

# 1. Introduction
Domain adaptation (DA) is essential for adapting machine learning models to diverse real-world scenarios with varying data distributions. These variations often require significant annotation effort, hindering model deployment. Unsupervised domain adaptation (UDA) offers a cost-effective solution by leveraging unlabeled target domain data, but it often underperforms compared to fully supervised learning. Active learning (AL) addresses this limitation by strategically selecting informative samples for labeling, enabling high accuracy with minimal annotation effort. However, traditional AL methods struggle with domain shift, where the source and target data distributions diverge. Active domain adaptation (ADA) combines domain adaptation with active sampling to enhance model performance under domain shift. While extensive research has explored ADA for classification and segmentation tasks, its application to object detection presents unique challenges, particularly in accurately predicting both object localization and category.

## 1.1. Paper Summary

This paper introduces a novel ADA method specifically tailored for object detection, addressing the limitations of conventional UDA and AL approaches. The authors identify false negative (FN) errors—undetected objects—as a critical issue under domain shift, which existing methods often fail to adequately address. To tackle this, they propose a False Negative Prediction Module (FNPM) that predicts the likelihood of FN errors, incorporating this metric into the active sampling process alongside uncertainty and diversity. Their framework starts with UDA training to align features between the source and target domains, followed by active sampling guided by the FNPM's predictions. The final semi-supervised DA training phase incorporates a few labeled target domain samples and pseudo-labels generated from uncertainty-guided predictions. Experimental results demonstrate that this method achieves performance comparable to fully supervised learning while requiring significantly less labeling effort, setting a new benchmark for ADA in object detection.

# 2. The method, its implementation and my interpretation
<img width="1058" alt="image" src="https://github.com/user-attachments/assets/bcc555e4-5294-4148-a599-2198c4885c87">

The schema of the suggested framework

## 2.1. The original method

This method uses a combination of Unsupervised Domain Adaptation (UDA) and Active Learning (AL) to improve the detection model’s performance, particularly in terms of reducing False Negative (FN) errors, which are common when transferring models across domains.

**Key Steps of the Original Method:**

### 2.1.1. Model Initialization:
   The method begins by initializing the model parameters using both the labeled source domain data (DS) and the labeled target domain data (DT) in an unsupervised domain adaptation (UDA) setting. The model is trained to adapt features across domains using adversarial learning, where the student and teacher models are involved. The student model learns from both source and target domains, and a domain discriminator helps align the features from both domains so that they are indistinguishable. 

   - The student model (θs, φs) and teacher model (θt, φt) are initialized to have the same parameters, ensuring consistency in training.
   - The adversarial loss is applied during training to ensure that the feature representations from the source and target domains are aligned, improving domain generalization.

### 2.1.2. Active Sampling Based on Acquisition Function:
   Once the model is initialized, the next step is to perform active sampling within the target domain. This process involves selecting a subset of unlabeled target domain images that are most informative for improving the model’s performance. The selection is guided by an acquisition function, which is designed to identify which images should be labeled to improve the model.

   The acquisition function works by considering factors like uncertainty and undetectability of objects in the images. The images that are most uncertain (i.e., where the model has high prediction variance) and those where objects are difficult to detect (FN errors) are prioritized for labeling. These images are then labeled by a human annotator and added to the labeled target domain data (DLT).

   - The active learning cycle involves selecting the top-K most informative images based on the acquisition score, which integrates uncertainty and FN metrics.
   - After the images are labeled, they are incorporated into the labeled dataset (DLT), and the unlabeled target domain dataset (DUT) is updated accordingly.

  There are four metrics that are combined in the proposed methos:
    - Undetectability (measures how challenging a sample is for the detection model, with higher values indicating greater difficulty and informativeness; FNPM is utilized, and I will explain it in next subsections)
    - Localization Uncertainty (assesses bounding box coordinate variation using variational inference)
    - Classification Uncertainty (measures class prediction difficulty using entropy of class probabilities)
    - Diversity (evaluates sample representativeness of the target domain using the domain discriminator)

  These four metrics are normalized to prevent dominance by large-scale metrics into a final resulting metric.




### 2.1.3. Training with Labeled and Unlabeled Data:
   After each round of active learning, the model is retrained using both the labeled source data (DS) and the labeled target data (DLT), while also considering the remaining unlabeled target domain data (DUT). The model is trained in a semi-supervised manner, where the labeled target domain data and source data provide supervision, while the unlabeled data is used to further adapt the model using techniques like pseudo-labeling.

   - The training involves updating the student model (θs) using the new labeled data and updating the teacher model (θt) using a moving average technique, ensuring stability in the model’s predictions.
   - This process helps the model progressively improve its ability to detect objects in the target domain, especially in challenging scenarios where objects might be missed due to domain differences.

### 2.1.4. Iteration:
   The above steps (active sampling, labeling, and training) are repeated for a fixed number of rounds (five rounds in the paper).

### 2.1.5. FN Errors and the Role of Active Learning (FNPM):
   A critical observation in the original method is that False Negative (FN) errors are a significant issue in domain adaptation for object detection. Although traditional domain adaptation methods can reduce False Positive (FP) errors, FN errors often persist, especially under domain shift. The paper proposes to focus active learning on sampling images where FN errors are likely to occur. This is where the FN Prediction Module (FNPM) comes into play—it predicts the number of FN errors for each image, helping to prioritize samples that will lead to a reduction in FN errors.
   
<img width="528" alt="image" src="https://github.com/user-attachments/assets/732ef228-ea40-4c5f-8683-066f9d419fcb">


False Negative Prediction Module (FNPM), uses the output feature map from the detection model’s backbone, which is processed through global average pooling (GAP) and fully connected (FC) layers to produce FN predictions. Ground truth FN counts are derived from the detection model’s output by identifying ground truths not matched to predictions with a sufficient intersection over union (IoU) score. The FNPM is trained with a loss function designed to minimize prediction errors. However, since labeled target domain data is limited, the FNPM leverages domain-invariant features extracted from the adapted backbone of the detection model to make predictions for the target domain, thereby avoiding the need for additional parameters. Training stability is maintained by alternately updating the FNPM and the detection model. During active sampling, the FNPM parameters are updated while the detection model is frozen, ensuring effective optimization for both components without interference. This approach ensures that FN errors are effectively addressed while maintaining simplicity and stability in the training process.

## 2.2. Datasets and Implementation Details 

### 2.2.1. Datasets
For this research, tests were conducted using five datasets. These datasets have the following descriptions:
Cityscapes (C) is a popular dataset that was taken from in-car cameras in cities with clear skies. It contains 500 validation images and 2,975 training photos with excellent bounding box annotations made from instance segmentation masks. One can directly utilize https://www.cityscapes-dataset.com/ to access the basic base dataset. There is also the option to view the dataset at https://www.kaggle.com/datasets/shuvoalok/cityscapes/data .
By adding artificial fog to the Cityscapes photos at varying densities (0.005, 0.01, and 0.02), Foggy Cityscapes (F) replicates unfavorable weather conditions. The highest fog density (0.02) is utilized for the paper's rating. Testing transition from clear to foggy surroundings is a difficult task. https://www.cityscapes- dataset.com/downloads/ is a link to download this dataset again. Furthermore, the following dataset is available on Kaggle: https://www.kaggle.com/datasets/khitdon/foggy-cityscapes-yolo
The extensive dataset BDD100k (B) includes a variety of driving sequences that were recorded in a range of settings, such as shifting weather and lighting. In order to assess adaptation to more varied and intricate real-world circumstances, we selected a daylight subset of this dataset that included 5,258 validation photos with bounding box annotations and 36,728 training images. The primary resource link for viewing this dataset is https://bdd-data.berkeley.edu/. https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k? select=bdd100k_labels_release is the link to it on Kaggle.
A synthetic dataset called SIM10k (S) was created with a video game engine and includes 10,000 images with 58,071 vehicle bounding boxes tagged on them. It emphasizes the difficulty of transferring information from virtual to real-world situations by acting as a source domain in synthetic-to-real adaption scenarios. This dataset can be directly accessed at https://fcav.engin.umich.edu/projects/driving-in-the-matrix.
Finally, KITTI (K) is an additional real-world dataset that depicts suburban and rural landscapes in contrast to Cityscapes' urban emphasis. We tested the capacity to generalize to new real-world circumstances using 7,481 photos from KITTI. In 2012 dataset used in the research "Are we ready for Autonomous Driving? KITTI Vision Benchmark Suite." is available in kaggle https://www.kaggle.com/datasets/klemenko/kitti-dataset

### 2.2.2. Implementation Details
For the experimental design part step by step we will implement those methodologies as below:
Here is the specifics of the Experiment's Implementation The architecture of networks:
In order to combine region proposal and object detection into a single framework, we used the Faster R-CNN detection model. We employed a BatchNorm-free VGG16 pre-trained on ImageNet as the backbone, in accordance with the implementation in (3). In order to effectively distinguish between source and target domain features, we used the architecture suggested in 17 for the domain discriminator.
Pre-processing
We adjusted each image's dimensions so that the shorter side was 600 pixels while keeping the aspect ratio constant in order to maintain the original proportions before feeding them into the model. We used both strong and weak augmentations on the images, as described in (3), to improve robustness.
  - Optimization:
      - Detection Model Training:
        Stochastic gradient descent (SGD) with a momentum of 0.9 will be used for optimization.To avoid overfitting, a weight decay of 10^-4 was         applied.Using a warm-up technique, the learning rate was progressively changed from its initial setting of 0.02. The learning rate was            lowered by a factor of 10 at 30000 and 35000 iterations during the 40000 iterations of training.
      - False Negative Prediction Module (FNPM):
        SGD optimization with a 10^-4 initial learning rate. A cosine annealing scheduler was used to train for 2000 iterations, gradually lowering       the learning rate.
  - Training Configuration
      - To guarantee a balanced sampling from each domain, the batch size for both source and target domain data was set at 4.
      - Five rounds of active sampling were conducted at 5k, 10k, 15k, 20k, and 25k iterations. The target samples that provided the most information     were chosen for annotation at the end of each round.
    - Key hyperparameters:
        - Variance threshold (𝛾) : 0.1
        - Dropout rate (𝜂): 0.1
        - Number of variational inferences (𝑀): 10 Exponential moving average (EMA) rate (𝛼): 0.9996 
        - Weight of adversarial loss (𝜆): 0.01
  - Evaluation Metrics:
  The average precision (AP) metric at an intersection over union (IoU) threshold of 0.5 was used to assess the effectiveness of our               approach. For multi-class scenarios, such as Cityscapes to BDD100k (C -> B) and Cityscapes to Foggy Cityscapes (C -> F), we reported the           mean average precision (mAP) for each object class.



## 2.3. My Interpretations

### 2.3.1. FNPM Implementation

Network Architecture: The FNPM can be implemented as a separate network or integrated into the object detection model. The architecture should be designed to capture both global and local image features relevant to object detection.
Training: The FNPM can be trained using a combination of supervised and unsupervised learning. Supervised training can be performed using a dataset of images with ground truth false negative annotations. Unsupervised training can leverage techniques like self-training or contrastive learning to learn representations that are robust to domain shifts.
### 2.3.2. Acquisition Function Tuning
The balance between uncertainty-based sampling and FNPM scores in the acquisition function can be tuned using techniques like hyperparameter optimization. Different weights can be assigned to the two components based on the specific domain shift and the desired level of exploration vs. exploitation.

### 2.3.3. Pseudo-Labeling Threshold
The threshold for selecting high-confidence pseudo-labels should be carefully chosen to balance the trade-off between increasing training data and introducing noise. A higher threshold can reduce noise but may also limit the amount of additional training data.

### 2.3.4. Computational Cost
The proposed method involves additional computational overhead due to the FNPM and uncertainty estimation. To mitigate this, efficient implementations and hardware acceleration techniques can be employed. Additionally, techniques like knowledge distillation can be used to compress the model and reduce inference time.

### 2.3.5. Domain Gap
The effectiveness of the method depends on the degree of domain gap between the source and target domains. If the gap is significant, additional techniques like domain adaptation or transfer learning may be necessary to improve performance.

### 2.3.5 Augmentations
The paper mentions several actions that can be taken as strong augmentation but does not clearly provide exact actions so, we implemented a strong data augmentation pipeline inspired by methods described in the referenced paper. The augmentation includes techniques like resizing with aspect ratio preservation, color jittering, random grayscale application, Gaussian blur, and solarization.

# 3. Experiments and results
We implemented the supervised student model of the framework, we yet have to dive into active sampling and unsupervised domain adaptation parts of the framework.
Note: At this phase we could not manage to run successfuly the training process with high number of iterations and epochs. That is why we have gotten onderfitted model with flactuating training and validation loss. 
## 3.1. Experimental setup
The implementation that is followed by the paper runs 30k epoch but as we do not have enough computational power for this process for initial phase we did applied 100 epochs.
For our implementation, we followed the training setup described in the paper for both the detection model and the False Negative Prediction Module (FNPM). For the detection models, we used Stochastic Gradient Descent (SGD) with a momentum of 0.9, a weight decay of \(10<sup>-4</sup>\), and started with a learning rate of 0.02. We also included a warm-up phase at the beginning of training. The model was trained for 40,000 iterations, and the learning rate was reduced by a factor of 10 at the 30,000 and 35,000 iteration marks. For the FNPM, we used a separate SGD optimizer with a smaller starting learning rate of \(10<sup>-4</sup>\). This part of the training lasted for 2,000 iterations, with a cosine annealing scheduler to gradually adjust the learning rate. We used this scheduler before starting active sampling to make sure the FNPM was optimized independently, avoiding any interference with the detection model.

## 3.2. Running the code
At first you need to install the datasets; For the first phase we only used Cityscapes dataset, and put it into Datasets folder under the main project folder with the same level with Codes folder. Afterwards, we run the function written in Collector module to reorganize the images under the datasets in a way that collects images from different cities(folders) into train/all_data, val/all_data and test/all_data. 
In the Codes folder we add Model_Results folder where we keep track and wights of training process and this let us add up new epoch runs instead of starting from scratch. 

## 3.3. Results


The training process for the Faster R-CNN model with a VGG16 backbone did not go as expected. From the graph of training and validation losses, it is clear that the model struggled to learn properly. Both the training and validation losses fluctuate a lot throughout the 50 epochs, and there is no clear sign that the losses are decreasing steadily. This indicates that the training process failed to converge, meaning the model did not successfully optimize its performance.

The validation loss is often higher than the training loss, which suggests that the model may be overfitting to the training data and not generalizing well to new data. The loss function used by Faster R-CNN combines different components like classification loss, regression loss, and region proposal network (RPN) loss. However, the way the losses behaved shows that the model was not able to minimize them effectively.

This unstable behavior could be caused by several issues, such as a learning rate that is too high, noisy or incorrect data, or problems in how the model was set up or trained. To fix this, adjustments like fine-tuning the learning rate, improving the data quality, or checking the annotations might help. For now, the results show that the training process was unsuccessful and needs improvement.

![image](https://github.com/user-attachments/assets/7b036eac-b55b-493f-9045-37fe46f8708f)


# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
