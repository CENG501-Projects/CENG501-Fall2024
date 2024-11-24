# MonoATT: Online Monocular 3D Object Detection with Adaptive Token Transformer

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).

## 1.1. Paper summary

MonoATT is an online Mono3D framework designed for accurate 3D object detection from a single camera input. It specifically targets mobile devices where computational power is limited and response time is critical, such as in autonomous driving applications. The rising success of transformers in NLP in the recent years has also sparked a trend of trials to integrate them into existing visual domains such as 3D object detection. Similar to the other Mono3D methods, MonoATT utilizes transformers but also a custom multi-step procedure.

Traditionally, Mono3D frameworks employ homogeneous grid-based vision tokens which presented two main issues: If a coarse grid is used, distant and small objects cannot be detected accurately enough. On the contrary, if a fine grid is used, increased computational complexity makes it quite a challenge to run on mobile applications. Recent works such as MonoDTR [x] and MonoDETR [x] have shown significant progress in monocular 3D object detection; however, their utilization of homogeneous grids leads to the mentioned problem. MonoATT addresses these challenges with a novel approach called Adaptive Token Transformer (ATT) by using heterogeneous tokens. This method assigns finer tokens to regions of the image that contain critical information (such as cars, pedestrians and bicycles), and coarser tokens to less important areas (sky, buildings and so on). This token distribution is based on these "keypoints", which are dynamically identified and prioritized by the framework.

![figure1](https://github.com/user-attachments/assets/25c72296-9f7e-42d0-a738-933a2bb99905)
<p align="center"> Figure 1. Grid-based approach of traditional Mono3D methods, and the heterogenous token approach of MonoATT. </p>


A multi-stage process is designed to implement this approach. First, a scoring network assesses parts of images to identify keypoints. These scores are determined by semantic information (importance for the context) and depth estimation (to emphasize far objects more than near ones). The results of the network is used for the subsequent clustering of tokens. Secondly, these tokens are merged using an attention mechanism that considers both feature similarities and spatial relationships to optimize the allocation of computational resources and improve detection accuracy. Finally, a pixel-level "enhanced" feature map is reconstructed from the heterogeneous tokens for effective and accurate 3D object detection. This feature map is rebuilt and upsampled so that it can work with existing Mono3D detectors, such as GUPNet [x].

Experimental results on the KITTI 3D dataset demonstrate that MonoATT significantly outperforms existing methods, offering improvements in detection accuracy for both near and far objects while maintaining low latency suitable for real-time applications. The paper concludes by acknowledging the framework's potential for further optimization, particularly in refining the efficiency of the token clustering process and improving the robustness of the scoring network against semantic noise.


# 2. The method and our interpretation

## 2.1. The original method

![figure2](https://github.com/user-attachments/assets/d394a131-5bf8-4add-935d-5257bdfebf04)

<p align="center"> Figure 2. Top view archtitecture of the MonoATT. </p>
MonoATT consists of different parts. In particular, there are 4 modules. Cluster center estimation (CCE), Adaptive Token Transformer (ATT), Multistage Feature Reconstruction (MFR), and monocular 3D detection. The image is not directly fed to the network. As a first step, a feature map is created by using DLA-34 as backbone.

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

$x_{3d} = \frac{u - c_x}{f_x} \hat{z}, \quad y_{3d} = \frac{v - c_y}{f_y} \hat{z}$

f_x and f_y represent the focal lengths in pixels along the axes and c_x and c_y are the possible displacement between the image center and the foot point. All those parameters are included in the camera intrinsic parameters which is called K. From the Equation 2:

$z = \frac{f_y \times y}{v - c_y}$

If elevation of the camera from the ground is assumed to be known and called H, the equation becomes:

$z = \frac{f_y * H}{v - c_y}$

Note that mean height of all vehicles in the KITTI dataset is 1.65m. Since the equation 3 is not continuous and the result may be negative, the following depth scoring function is used:

$S_d = -\text{ReLU}\left(B \frac{v - c_y}{f_y \cdot H}\right)$ , where B is a constant. 

#### 2.1.1.2 Semantic Scoring Function
For the semantic scoring function, a neural network is used to detect the keypoints from the images. A regression branch is added to the CenterNet. 
    $S_s = f(H)$ where H is the input image feature obtained from the DLA-34 and f is the CNN architecture. 

Total score is calculated as;
    $S = S_d + \alpha S_s$ where $\alpha$ is a hyperparameter.

The loss of point detection is calculated as;

$\mathcal{L}_{\text{CCE}} = \text{FL}\left(\mathbf{g}^m(\mathbf{u}_t, \mathbf{v}_t), \mathbf{S}\right)$

where FL is the Focal Loss, $(\mathbf{u}_t, \mathbf{v}_t)$ is the ground truth key point coordinate, $\mathbf{g}^m$ is the mapping function which turns m point coordinates into heatmap.

After scoring is done for the whole feature map, mean value of each pixel score within a token is taken as the importance of the token. For the token clustering, a cluster center token is chosen which has the highest average score. In some stages, there may be more than one cluster center required. In that case, at each iteration, nth cluster center is chosen among the clusters which has the highest ranking.

### 2.1.2 Adaptive Token Transformer (ATT)
By using both the tokents from the initial stage (feature map) and selected cluster centers, tokens are grouped into clusters. Tokens in each cluster are merged and single token for every cluster is created. Then results are fed into a transformer. Adaptive token transformer exploits the long-range self-attention mechanism. 

![figure3](https://github.com/user-attachments/assets/1f395206-1392-40a2-8c27-239376c4d4bc)
<p align="center"> Figure 3. ATT loops N stages where at every stage first outline preferred token grouipng and then later, attention based feature merging are performed.  </p>

#### 2.1.2.1 Outline-preferred Token Grouping
To avoid local feature correlation, clustering based on the spatial distance is not reasonable. A variation of nearest neighbor algorithm is used. The algorithm considers feature similarity along with the image distance while clustering. 

$\delta_i = \min_{j : \mathbf{x}_j \in \mathbf{X}_c} \left( \lVert \mathbf{x}_i - \mathbf{x}_j \rVert_2^2 - \beta \lVert \mathbf{g}^l(\mathbf{x}_i) - \mathbf{g}^l(\mathbf{x}_j) \rVert_2^2 \right)$

The equation states that, for every point each cluster center is looped. Feature similarity and spatial distances are considered and balance between them is controlled by \Beta. Minimum value of a point against a cluster means that, the most appropriate value for the cluster is found.

#### 2.1.2.2 Attention-based Feature Merging
While merging tokens, instead of directly averaging the token features at every cluster, token features are averaged with the guidance of attention scores. 

$\mathbf{y}\_i = \frac{\sum_{j \in \mathcal{C}\_i} e^{p\_j} \mathbf{x}\_j}{\sum_{j \in \mathcal{C}\_i} e^{p\_j}}$ 

where $\mathbf{y}_i$ is the merged token feature, $\mathbf{x}_j$ and $p_j$ are the original token features and they are summed over the i'th cluster set. 
Merged tokens are fed into the transformer as queries Q. Original tokens are also used as keys K and values V. Attention matrix of the transformer altered and attention score p is involved in the calculation in order to allow tokens to be more important. 

$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{p}\right) \mathbf{V}$ where $d_k$ is the number of the channels of the queries. 

Token Features might have dimensions (n,d), where n is the number of tokens and d is the feature dimension. Attention Scores might have a dimension of
(n,1), as each token gets a single score. To combine these (e.g., by addition), their shapes must match. If they don’t, matrix expansion ensures that smaller matrices (like (n,1)) are "stretched" to match larger ones (like (n,d)). Also, token attention score allows network to focus on the important information.

### 2.1.3 Multi-stage Feature Reconstruction (MFR)
Multi-stage Feature Reconstruction restores and aggregates all N stages. MFR upsamples the tokens from a history record and restores the feature map.

![figure4](https://github.com/user-attachments/assets/d282ae2f-a2ce-4f79-b364-40ae1c9372b7)
<p align="center"> Figure 4. Upsampling and reconstruction of feature map.  </p>

In Attention-based Feature Merging every token is assigned to a cluster and since tokens are merged, every cluster is represented by a merged token. Positional correspondences between original and merged tokens are recorded and those records are used to copy the merged token features into the unsampled token. Token features from the previous stage are added iteratively to get an aggregate. Later, tokens are fed to a multilayer perceptron and that process is iterated N times. After the process a feature map of size $(W_s \times H_s \times C')$ is obtained.

### 2.1.4 Monocular 3D Detection
In MonoATT, GUPNet is used as a monocular 3D object detector.

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
