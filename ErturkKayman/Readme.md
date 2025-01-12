# MonoATT: Online Monocular 3D Object Detection with Adaptive Token Transformer

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper selected for our course project, "MonoATT: Online Monocular 3D Object Detection with Adaptive Token Transformer" [[1](https://arxiv.org/abs/2303.13018)], is an online Mono3D framework authored by Yunsong Zhou, Hongzi Zhu, Quan Liu, Shan Chang, and Minyi Guo. This paper was presented at the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023. Our goal is to reproduce the findings of this paper by implementing the proposed theoretical framework and methods as described, including the detailed formulas provided. We aim to develop and integrate all four modules to function as explained in the paper and get similar performance to the benchmark results given.

## 1.1. Paper summary

MonoATT is an online Mono3D framework designed for accurate 3D object detection from a single camera input. It specifically targets mobile devices where computational power is limited and response time is critical, such as in autonomous driving applications. The rising success of transformers in NLP in the recent years has also sparked a trend of trials to integrate them into existing visual domains such as 3D object detection. Similar to the other Mono3D methods, MonoATT utilizes transformers but also a custom multi-step procedure.

Traditionally, Mono3D frameworks employ homogeneous grid-based vision tokens which presented two main issues: If a coarse grid is used, distant and small objects cannot be detected accurately enough. On the contrary, if a fine grid is used, increased computational complexity makes it quite a challenge to run on mobile applications. Recent works such as MonoDTR [[2](https://arxiv.org/abs/2203.10981)] and MonoDETR [[3](https://arxiv.org/abs/2203.13310)] have shown significant progress in monocular 3D object detection; however, their utilization of homogeneous grids leads to the mentioned problem. MonoATT addresses these challenges with a novel approach called Adaptive Token Transformer (ATT) by using heterogeneous tokens. This method assigns finer tokens to regions of the image that contain critical information (such as cars, pedestrians and bicycles), and coarser tokens to less important areas (sky, buildings and so on). This token distribution is based on these "keypoints", which are dynamically identified and prioritized by the framework.


<p align="center"> 
<img src="https://github.com/user-attachments/assets/25c72296-9f7e-42d0-a738-933a2bb99905">      
Figure 1. Grid-based approach of traditional Mono3D methods, and the heterogenous token approach of MonoATT. </p>


A multi-stage process is designed to implement this approach. First, a scoring network assesses parts of images to identify keypoints. These scores are determined by semantic information (importance for the context) and depth estimation (to emphasize far objects more than near ones). The results of the network is used for the subsequent clustering of tokens. Secondly, these tokens are merged using an attention mechanism that considers both feature similarities and spatial relationships to optimize the allocation of computational resources and improve detection accuracy. Finally, a pixel-level "enhanced" feature map is reconstructed from the heterogeneous tokens for effective and accurate 3D object detection. This feature map is rebuilt and upsampled so that it can work with existing Mono3D detectors, such as GUPNet [x].

Experimental results on the KITTI 3D dataset demonstrate that MonoATT significantly outperforms existing methods, offering improvements in detection accuracy for both near and far objects while maintaining low latency suitable for real-time applications. The paper concludes by acknowledging the framework's potential for further optimization, particularly in refining the efficiency of the token clustering process and improving the robustness of the scoring network against semantic noise.


# 2. The method and our interpretation

## 2.1. The original method

MonoATT consists of different parts. In particular, there are 4 modules. Cluster center estimation (CCE), Adaptive Token Transformer (ATT), Multistage Feature Reconstruction (MFR), and monocular 3D detection. The image is not directly fed to the network. As a first step, a feature map is created by using DLA-34 as backbone.

<p align="center">
<img src="https://github.com/user-attachments/assets/d394a131-5bf8-4add-935d-5257bdfebf04">     
Figure 2. Top view archtitecture of the MonoATT. </p>

MonoATT uses adaptive tokens with irregular shapes and various sizes in order to accomplish two goals:
1) Increasing the accuracy of both near and far objects by obtaining superior image features.
2) Improving the time performance of the visual transformer by reducing the number of tokens in the irrelevant regions.

First step of the architecture is to create a feature map by using DLA-34 as a backbone. From a monocular image which has the dimensions $(W \times H \times 3)$, a feature map with dimensions $(W_s \times H_s \times C)$ is obtained where S is a hyperparameter. Then, the feature map is fed into the following modules:
### 2.1.1 Cluster Center Estimation (CCE)

Cluster center estimation module decides the cluster centers based on the importance of the coordinates. Each region has different importance and two facts are considered to decide the importance of the region. 
1) As distant objects are harder to detect, they should receive more attention.
2) As a semantic knowledge, features belong to the target classes are more important than the non-target classes such as background. In addition; corners, boundries, etc. are more important than the inner features of the target.

To comply with the both observations, two scoring functions are purposed which are called depth scoring function and semantic scoring function. 

#### 2.1.1.1 Depth Scoring Function
By using the pinhole camera model, in a given the camera coordinate system P, the virtual horizontal plane can be projected on the image
plane of the camera, and by using camera instrinsic parameter K, depth of each pixel can be determined. Every point in 2D scene with the coordinates u,v can be projected to the 3D scene with x,y,z coordinates as follows;

<p align="center">
$x_{3d} = \frac{u - c_x}{f_x} \hat{z}, \quad y_{3d} = \frac{v - c_y}{f_y} \hat{z}$
</p>

f_x and f_y represent the focal lengths in pixels along the axes and c_x and c_y are the possible displacement between the image center and the foot point. All those parameters are included in the camera intrinsic parameters which is called K. From the Equation 2:

<p align="center">
$z = \frac{f_y \times y}{v - c_y}$
</p>

If elevation of the camera from the ground is assumed to be known and called H, the equation becomes:

<p align="center">
$z = \frac{f_y * H}{v - c_y}$
</p>

Note that mean height of all vehicles in the KITTI dataset is 1.65m. Since the equation 3 is not continuous and the result may be negative, the following depth scoring function is used:

<p align="center">
$S_d = -\text{ReLU}\left(B \frac{v - c_y}{f_y \cdot H}\right)$
</p>

where B is a constant. 

#### 2.1.1.2 Semantic Scoring Function
For the semantic scoring function, a neural network is used to detect the keypoints from the images. A regression branch is added to the CenterNet. 
    $S_s = f(H)$ where H is the input image feature obtained from the DLA-34 and f is the CNN architecture. 

Total score is calculated as;
    $S = S_d + \alpha S_s$ where $\alpha$ is a hyperparameter.

The loss of point detection is calculated as;

<p align="center">
$\mathcal{L}_{\text{CCE}} = \text{FL}\left(\mathbf{g}^m(\mathbf{u}_t, \mathbf{v}_t), \mathbf{S}\right)$
</p>

where FL is the Focal Loss, $(\mathbf{u}_t, \mathbf{v}_t)$ is the ground truth key point coordinate, $\mathbf{g}^m$ is the mapping function which turns m point coordinates into heatmap. After scoring is done for the whole feature map, mean value of each pixel score within a token is taken as the importance of the token. For the token clustering, a cluster center token is chosen which has the highest average score. In some stages, there may be more than one cluster center required. In that case, at each iteration, nth cluster center is chosen among the clusters which has the highest ranking.

### 2.1.2 Adaptive Token Transformer (ATT)
By using both the tokents from the initial stage (feature map) and selected cluster centers, tokens are grouped into clusters. Tokens in each cluster are merged and single token for every cluster is created. Then results are fed into a transformer. Adaptive token transformer exploits the long-range self-attention mechanism. 


<p align="center"> 
<img src="https://github.com/user-attachments/assets/1f395206-1392-40a2-8c27-239376c4d4bc">    
Figure 3. ATT loops N stages where at every stage first outline preferred token grouipng and then later, attention based feature merging are performed.  </p>


#### 2.1.2.1 Outline-preferred Token Grouping
To avoid local feature correlation, clustering based on the spatial distance is not reasonable. A variation of nearest neighbor algorithm is used. The algorithm considers feature similarity along with the image distance while clustering. 

<p align="center">
$\delta_i = \min_{j : \mathbf{x}_j \in \mathbf{X}_c} \left( \lVert \mathbf{x}_i - \mathbf{x}_j \rVert_2^2 - \beta \lVert \mathbf{g}^l(\mathbf{x}_i) - \mathbf{g}^l(\mathbf{x}_j) \rVert_2^2 \right)$
</p>

The equation states that, for every point each cluster center is looped. Feature similarity and spatial distances are considered and balance between them is controlled by $\beta$. Minimum value of a point against a cluster means that, the most appropriate value for the cluster is found.

#### 2.1.2.2 Attention-based Feature Merging
While merging tokens, instead of directly averaging the token features at every cluster, token features are averaged with the guidance of attention scores. 

<p align="center">
$\mathbf{y}\_i = \frac{\sum_{j \in \mathcal{C}\_i} e^{p\_j} \mathbf{x}\_j}{\sum_{j \in \mathcal{C}\_i} e^{p\_j}}$ 
</p>

where $\mathbf{y}_i$ is the merged token feature, $\mathbf{x}_j$ and $p_j$ are the original token features and they are summed over the i'th cluster set. 
Merged tokens are fed into the transformer as queries Q. Original tokens are also used as keys K and values V. Attention matrix of the transformer altered and attention score p is involved in the calculation in order to allow tokens to be more important. 

<p align="center">
$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{p}\right) \mathbf{V}$
</p>

where $d_k$ is the number of the channels of the queries. Token Features might have dimensions (n,d), where n is the number of tokens and d is the feature dimension. Attention Scores might have a dimension of
(n,1), as each token gets a single score. To combine these (e.g., by addition), their shapes must match. If they don’t, matrix expansion ensures that smaller matrices (like (n,1)) are "stretched" to match larger ones (like (n,d)). Also, token attention score allows network to focus on the important information.

### 2.1.3 Multi-stage Feature Reconstruction (MFR)
Multi-stage Feature Reconstruction restores and aggregates all N stages. MFR upsamples the tokens from a history record and restores the feature map.


<p align="center"> 
<img src="https://github.com/user-attachments/assets/d282ae2f-a2ce-4f79-b364-40ae1c9372b7">    
Figure 4. Upsampling and reconstruction of feature map.  </p>


In Attention-based Feature Merging every token is assigned to a cluster and since tokens are merged, every cluster is represented by a merged token. Positional correspondences between original and merged tokens are recorded and those records are used to copy the merged token features into the unsampled token. Token features from the previous stage are added iteratively to get an aggregate. Later, tokens are fed to a multilayer perceptron and that process is iterated N times. After the process a feature map of size $(W_s \times H_s \times C')$ is obtained.

### 2.1.4 Monocular 3D Detection
In MonoATT, GUPNet is used as a monocular 3D object detector.

## 2.2. Our interpretation

The paper provides a new perspective for vision transformers. Especially, by introducing some new learnable concepts, they further increase the capabilities of a vision transformer model. However, some points are unclear in the paper. 

Though paper is about avoiding grid based square tokents, it is not clear the initial grid size. Using every pixel as a grid is hugely expensive even though the paper uses strategies like token merging. The paper aims to create irregular shaped image patches. However, initially, we had to adopt 4x4 patches before merging clustering patches due to complexity of the architecture.

While extracting the feature map, s-factor is unclear. The writers have used 5 as s-factor in their previous paper. We adopted the backbone from their previous paper called MonoDETR since finetuning backbone is not computationly feasible and it is usualy similar in 3D object detection tasks.

In the cluster center estimation part,it is not clear which CNN architecture is used for semantic scoring. It is only mentioned that it is a regression branch from CenterNet. Currently, we did not have time to try different CNN architectures. We have used the simple network below for semantic scoring after some research. We did not have time to try different semantic scoring architectures.
```python
# Heatmap head for semantic scoring
self.heatmap_head = nn.Sequential(
    nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=True),  #I am forced to use 256 because of complexity.
    nn.GroupNorm(32, 256),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 1, kernel_size=1, bias=True),  # Output single-channel heatmap
)
```

In the Adaptive Token Transformer part, the number of loops N is unclear. After some research and trials, we have decided to make it generic so that we loop until all tokens are aggregated. Also a very important hyperparameter number of clusters is unknown. We have decided it to be 100 after a few trials. Also none of the transformer parameters are known. Though we adapt some of them from authors previous work, we had to use some lightweight values to minimize the computational costs.

In the Outline-preferred Token Grouping part, hyperparameter B is unclear. We did not have a chance to fintune B value and we used the value suggested by ChatGPT which is 1.

In Multi-stage Feature Reconstruction part, the number of loops N is unclear. We implemented MFR as it aggregates all tokens in a single step for now.

The architecture is really complex and there are many more hyperparameters which are unclear. Due to the complexity of the architecture, hyperparameter search methods such as grid search etc. are not feasible. We have adopted most of the hyperparameters from the authors previous work which is also about monocular 3D detection. But there is no guarantee that they remain the same. Also, other than the very major hyperparameters mentioned above, there are many different hyperparameters on the novel parts of the paper which we had to use trial and error as much as possible.

# 3. Experiments and results

## 3.1. Experimental setup

It is not easy to implement and finetune such a huge network from stracth. Hence, we decided to adapt the previous paper of the authors called MonoDetr. MonoDetr already implement the KITTI Dataset loader, trainer, tester, many of the loss calculations we need in this paper. We switched MonoDetr with MonoATT, added necessary new models such as AdaptiveTokenClustering, ClusterCenterEstimation, added related losses mentioned in the paper and trained. 

Though we tried to stick the MonoATT paper; due to time constraints, implementation difficulties and huge number of hyperparameters, we could not follow the paper exactly on some points. The table below summarizes our implementation vs. paper's implementation. 

| Aspect                  | MonoATT Paper               | Our Implementation         | Status/Comments                        |
|-------------------------|-----------------------|----------------------------|----------------------------------------|
| *Backbone*            | DLA-34               | ResNet-50                  | We got worse results with DLA-34, hence kept the ResNet-50 from MonoDetr                  |
| *Multi-stage Feature Reconstruction (MFR)* | Global/local integration    | Single-stage reconstruction            | Basic, functional                  |
| *Detection Head*      | GUPNet               | Custom detection head       | We kept the detection head from MonoDetr                     |
| *Loss Function*       | Composite losses      | Smooth L1 loss              | Needs refinement                       |
| *Evaluation*          | KITTI metrics        | Basic evaluation scripts    | Needs refinement                      |

We have used the libraries and tools:
  - *PyTorch:* Framework for model development and training.
  - *Torchvision:* For pre-trained backbones (e.g., ResNet).
  - *NumPy & Pandas:* For dataset manipulation and numerical operations.
  - *Tqdm:" For progress bar
  - *KITTI Dataset:* Used for training and evaluation.
  - *Matplotlib:* For visualizing results (predictions and bounding boxes).
  - *Deformable DETR:*  Deformable DETR is an efficient and fast-converging end-to-end object detector. (Which requires special compilation for each GPU-Cuda pair.)

    
We have implemented (fully or partially):
- Dataset Loading:
    Parsed the KITTI dataset to load images and their corresponding 3D bounding box annotations.
    Applied transformations (e.g., resizing, normalization) to preprocess the images.
    Handled variable-sized bounding boxes using a custom collate function.
    (Mostly adopted from MonoDetr.)
- Backbone Feature Extraction:
    Used ResNet-50 (pre-trained) as the feature extractor to generate a low-resolution feature map from input images.
    Our implementation outputs 4 different feature layers.
    Example output: Feature maps of shape (B, 512, H, W).
    (Mostly adopted from MonoDetr.)
- Cluster Center Estimation:
    From the features provided by backbone, semantic scores and depth scores are calculated and cluster  centers are estimated.
    Output: Returns combinaton of scores for loss calculations, final cluster centers ([B, num_clusters, C]), token positions (Shape: [B, num_tokens, 2]) and tokens ([B, num_tokens, C]).
- Adaptive Token Transformer:
    Flattened the feature map into tokens and clustered them using outline-preferred token grouping to identify regions of interest.
    Applied a single-layer attention mechanism to refine these tokens. Transformer is as small as possible due to computational challenges. 
    Output: Cluster assingments (Shape: [num_tokens]) and merged features. (Shape: [B, num_clusters, token_dim])
- Multi-stage Feature Reconstruction (Simplified):
    Used a fully connected layer and convolutional layers to reconstruct the pixel-level feature map from refined tokens.
    Added skip connections to preserve original spatial information.
    Output: Reconstructed feature maps of shape (B, C, H, W).
    (Mostly adopted from MonoDetr, not fully implemented because of the unknown hyperparameters. Instead we integrated the MonoDetr's output mechanism.)
- Mono3D Detection Head:
    Designed a detection head to predict 3D bounding box parameters:
        Location: (x, y, z)
        Dimensions: (h, w, l)
        Orientation: (theta)
    Output: Tensor of shape (B, 7, H’, W’).
- Loss Function:
    On top of the loss from MonoDetr for 3D object detection, loss used for scoring is added. It is implemented as a focal loss as mentioned in the paper. 
- Training:
    Trained the model on the KITTI dataset for 200 epochs.
    Training parameters are in the config file. 
- Evaluation:
    During training after every epoch (can be adjusted in the config) an evaluation is performed. 
    AP@40 scores are reported. (See next chapter for detailed explanation)

### 3.1.1. Acknowledgments

As mentioned earlier, we did not implement the network from stracth. We have used software from many different resources. Most of the archtiecture, even file structure, is taken from [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR) implementation. Because that paper has the exact same goal as ours (3D object detection on KITTI dataset), Dataset loader, trainer, tester, optimizer, losses are similar. We built our model on top of that codebase. We have also used codes directly from [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR), [DETR](https://github.com/facebookresearch/detr), [GUPNET](https://github.com/SuperMHP/GUPNet). 

Other than the implementation, for visualizing the evaluation results, we have used [KITTI Native Evaluation](https://github.com/asharakeh/kitti_native_evaluation).

In our file structure, here are the codes written (fully, partially) by us;

+ lib/models/monoatt/adaptive_token_clustering.py 
+ lib/models/monoatt/cluster_center_estimation.py
+ lib/models/monoatt/monoatt.py (partially)
+ lib/datasets/kitti/kitti_eval_python/kitti_common.py (partially)
+ lib/losses/focal_loss.py (partially)

## 3.2. Running the code
### 3.2.1 Training
- Create a new conda environment and activate.
```
  conda create -n monoatt python=3.8
  conda activate monoatt
```

- Make sure that pytorch and torchvision and cuda are installed. In our test we used;
  ```
  PyTorch 2.5.0
  Torchvision 0.20.0
  CUDA Version 12.2
  ```

- Compile Deformable Attention
```
    cd lib/models/monoatt/ops/
    bash make.sh
```

- Make dictionary for saving training losses:
    ```
    mkdir logs
    ```
- Download [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) datasets and prepare the directory structure;
  ```
    ├──...
    ├──MonoATT/
    ├──Dataset/KITTI/
    │   ├──ImageSets/
    │   ├──training/
    ├──...
    ```
  You can also change the data path at "dataset/root_dir" in `configs/monoatt.yaml`.

- Indicate the available GPUs in the train.sh. CUDA and GPU is a must, otherwise it won't work.
- Start the training with;
  ```
  bash train.sh configs/monoatt.yaml > logs/monoatt.log
  ```
  But as the architecture is complex and it will likely take a long time you may consider using;
  ```
  nohup bash train.sh configs/monoatt.yaml > logs/monoatt.log 2>&1 &
  ```
  Training process will evaluate the model after every epoch. (Configurable in the config file.) It will compare the existing the best with the current evaluation result and along with the latest checkpoint, it will also save the best checkpoint as well. 
### 3.2.2 Testing
- In order to test the checkpoint, run
  ```
  bash test.sh configs/monoatt.yaml
  ```
  The best checkpoint will be evaluated by default but it is configurable via config file. 

## 3.3. Results

The paper uses AP40 scores of the car category on KITTI test set at 0.7 IoU threshold to measure the performance of the model. It is also the official evaluation of KITTI contest. Results are available at the [KITTI website](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). 

### 3.3.1. AP40@0.7 Metric

The paper uses AP40 metric at 0.7 IoU threshold. The term AP40@0.7 is a performance metric commonly used in object detection tasks, particularly in 3D object detection tasks, such as those in autonomous driving. Let’s break it down:

*AP (Average Precision):*
- Average Precision (AP) evaluates the quality of an object detection model.
- AP measures the area under the Precision-Recall (PR) curve.
- Precision: The proportion of correctly identified objects (true positives) out of all predicted objects (true positives + false positives).
- Recall: The proportion of correctly identified objects out of all ground-truth objects.
- AP is a summary metric, combining precision and recall into a single number.

*40 in AP40:*
- The 40 here refers to the number of recall points used to calculate the metric.
- In AP40, the PR curve is sampled at 40 recall levels (e.g., 0.025, 0.05, ..., 1.0), and the average precision is computed as the mean precision at these points.
- This is in contrast to AP11, which uses only 11 recall points, and may be less granular. A few years ego, KITTI switched from AP11 to AP40.

*@0.7:*
The @0.7 specifies the Intersection over Union (IoU) threshold.
IoU is a measure of overlap between the predicted bounding box and the ground-truth bounding box:
“IoU\=Area of UnionArea of Overlap​”
 
For a prediction to be considered a true positive, the IoU between the predicted box and the ground truth must be at least 0.7.

### 3.3.2. MonoATT Paper Results 

According to the paper, MonoATT results are, 
<p align="center">
<img src=https://github.com/user-attachments/assets/0ea9c088-07e2-40e9-885b-3b3ff3b226bb>
Figure 8. Expected AP40 values table from the paper.
</p>
However, it is not possible for us to test our model on the test set. Test set labels are not shared and to get the results of the test set, one needs to submit the model to the KITTI website. However, we cannot submit our results to the contest because of the rules. KITTI website states that model submission step must only be followed if a paper is about to be submitted to a conference where the experimental results are ready, other evaluations (eg, in the context of model ablations, a student's class project or Master's thesis) must be conducted on the training set. Hence, we decided to split the training set into two equal parts for training and testing. 

In different repositories, there are some example training set splits for KITTI dataset. We have also provided our training-test split files under the dataset directory. However, MonoATT paper does not give any information about the training-test split. After some research, we have concluded that on monocular 3D detection tasks, usually splitting the KITTI training set which has 7500 images into half for training and test is a good practice. Our training-test split is taken from the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/tree/master/data/kitti) repository.

The paper also discusses the results on the validation set, which is some part of the training set (usually half in 3D object detection). They do provide an ablation study where we can see the contrubition of each component of the MonoATT archtitecture. At the bottom of the table, results for 3D object detection on validation set exists.

![image](https://github.com/user-attachments/assets/d9ceb45b-7663-4434-8579-8ff58a323323)

The paper also discusses integrating MonoATT components into existing transformer-based models. They claim to improve the AP40 scores of MonoDTR and MonoDETR structures. Validation set results are provided below. Please note that, as we used MonoDETR as base to implement MonoATT, it can be a good idea to compare our results with results provided in the table below.

![image](https://github.com/user-attachments/assets/e3d02550-5d80-42f0-a6e2-b274cc6aa980)

### 3.3.3 Our Test Results

We have provided a full training log (monoatt.log), prediction results and the best and the last trained models from our final training run. Please note that, during the training, the framework evaluates the model with the test test which can be found in the training logs, and saves the last and best models automatically. Provided outputs are the results of the best model. Our complete results are given in the table below.

Our results on Car category with the metric AP@0.70, 0.70, 0.70;
| Task  | Easy | Moderate | Hard |
| --- | --- | --- | ---| 
| bbox  | 90.4174  | 87.8160  | 79.9571 |
| bev  | 38.6704  | 30.3035  | 25.9506 |
| 3d  | 30.2722  | 24.3026  | 20.0706 |
| aos  | 89.12  | 84.95  | 76.66 |


Our results on Car category with the metric AP_R40@0.70, 0.70, 0.70 (which the paper uses to evaluate performance);
| Task  | Easy | Moderate | Hard |
| --- | --- | --- | ---| 
| bbox  | 96.2425  | 87.8052 | 80.5192 |
| bev  | 35.8728 | 25.9677  | 22.2793 |
| 3d  | 27.2766 | 19.5595  | 16.3184 |
| aos  | 94.66 | 84.85  | 76.99 |


Our results on Car category with the metric AP@0.70, 0.50, 0.50;
| Task  | Easy | Moderate | Hard |
| --- | --- | --- | ---| 
| bbox  | 90.4174 | 87.8160 | 79.9571 |
| bev  | 71.4474 | 53.3590  | 47.0313 |
| 3d  | 65.0951 | 47.8151  | 44.9623 |
| aos  | 89.12 |  84.95  | 76.66 |


Our results on Car category with the metric AP_R40@0.70, 0.50, 0.50:
| Task  | Easy | Moderate | Hard |
| --- | --- | --- | ---| 
| bbox  | 96.2425 | 87.8052 | 80.5192 |
| bev  | 70.5631 | 51.0103  | 45.9255 |
| 3d  | 67.0701 | 47.9734  | 42.8475 |
| aos  | 94.66 |  84.85  | 76.99 |

The results are segmented by different IoU thresholds (@0.70, 0.70, 0.70 and @0.70, 0.50, 0.50), which show the model's performance under varying levels of strictness for bounding box overlap.
Metrics evaluated include:
- bbox AP: Accuracy of 2D bounding box predictions.
- bev AP: Accuracy of bird's-eye view predictions.
- 3d AP: Accuracy of 3D bounding box predictions.
- aos AP: Orientation accuracy (alignment between predicted and actual orientations).
However, the research only minds 3d and bev results. Before comparing the results with the paper, it is safe to say on 3d tasks (3d and bev), model performs much much better if strict thresholds are not used. That indicates the model's capacity for accuractely detecting 3D objects, however, slight inaccuries on the bounding box placement or exact coordinate detection so that when the threshold is strict it causes the detection to fail.

When we compare our AP_R40@0.70, 0.70, 0.70 results with the paper, we can claim that we produced comparable but slightly bad results. As we inherit some parts from the MonoDETE architecture, it could be a good idea to compare our results with MonoDETR results where MonoATT is integrated which is given in the Image X. The comparison results show that for 3d detection tasks (3d and bev), for all difficulty levels, we produces approximately 2 points less than reported in the paper. Considering the unknown hyperparameters, lack of computation power and other results we will discuss in the conclusion, it can be considered as normal. 

We will also provide precision-recall curves for each task. Precision-recall curves can be used to calculate all the results we provided earlier. Also, for exact calculations, on the output directory, stats files exist for each task, where one can find the corresping precision point for each recall value from 0 to 1. For the details, please check the [kitti_eval](https://github.com/prclibo/kitti_eval) repository where we took the code to both calculate stats and draw graphs.

![car_detection](https://github.com/user-attachments/assets/28e6593e-c40f-4909-83c9-002a16cf53b0)

![car_detection_ground](https://github.com/user-attachments/assets/cc662b7f-71cf-41e2-b5d0-e055dd6fc26e)

![car_detection_3d](https://github.com/user-attachments/assets/8deedf7c-2018-4e7c-bdce-533c4797a913)

![car_orientation](https://github.com/user-attachments/assets/81250bd0-2be4-4e75-ad58-89a5a1a62d68)

In order to create the graphs or stats;

- Compile the kitti_eval project. (Check README of the project.)
- Then, use ./evaluate_object_3d_offline_ap40 TrainingDataset/label_2/ outputs/
- Results will be placed under the outputs directory.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

[1] Zhou, Y., Zhu, H., Liu, Q., Chang, S., & Guo, M. (2023, March 23). MonoATT: Online Monocular 3D Object Detection with Adaptive Token Transformer. arXiv.org. https://arxiv.org/abs/2303.13018

[2] Huang, K., Wu, T., Su, H., & Hsu, W. H. (2022, March 21). MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer. arXiv.org. https://arxiv.org/abs/2203.10981

[3] Zhang, R., Qiu, H., Wang, T., Guo, Z., Xu, X., Cui, Z., Qiao, Y., Gao, P., & Li, H. (2022, March 24). MONODETR: depth-guided transformer for monocular 3D object detection. arXiv.org. https://arxiv.org/abs/2203.13310

# Contact

- Çağatay Kayman - 1317kc@gmail.com
- Erdem Ertürk - erdemerturkk@gmail.com
