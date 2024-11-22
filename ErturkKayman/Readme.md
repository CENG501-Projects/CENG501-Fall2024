# MonoATT: Online Monocular 3D Object Detection with Adaptive Token Transformer

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and our interpretation

## 2.1. The original method

MonoATT consists of different parts. In spesific, there are 4 modules. Cluster center estimation (CCE), Adaptive Token Transformer (ATT), Multistage Feature Reconstruction (MFR), and monocular 3D detection. The image is not directly fed to the network. As a first step, a feature map is created by using DLA-34 as backbone. Top view archtitecture of the MonoATT is given in the Figure XX.

MonoATT uses adaptive tokens with irregular shapes and various sizes in order to accomplish two goals:
1) Increasing the accuracy of both near and far objects by obtaining superior image features.
2) Improving the time performence of the visual transformer by reducing the number of tokens in the irrelevant regions.

First step of the architecture is to create a feature map by using DLA-34 as a backbone. From a monocular image which has the dimensions W x H x 3, a feature map with dimensions Ws x Hs x C is obtained where S is a hyperparameter. Then, the feature map is fed into the following modules:
### 2.1.1 Cluster Center Estimation (CCE)

Cluster center estimation module decides the cluster centers based on the importance of the coordinates. Each region has different importance and two facts are considered to decide the importance of the region. 
1) As distant objects are harder to detect, they should receive more attention.
2) As a semantic knowledge, features belong to the target classes are more important than the non-target classes such as background. In addition; corners, boundries, etc. are more important than the inner features of the target.

To comply with the both observations, two scoring functions are purposed which are called depth scoring function and semantic scoring function. 

#### 2.1.1.1 Depth Scoring Function
By using the pinhole camera model, in a given the camera coordinate system P, the virtual horizontal plane can be projected on the image
plane of the camera, and by using camera instrinsic parameter K, depth of each pixel can be determined. Every point in 2D scene with the coordinates u,v can be projected to the 3D scene with x,y,z coordinates as follows;
    x = (u - c_x) * z / f_x ;
    y = (v - c_y) * z / f_y ;

f_x and f_y represent the focal lengths in pixels along the axes and c_x and c_y are the possible displacement between the image center and the foot point. All those parameters are included in the camera intrinsic parameters which is called K. From the equation 2;
    z = (f_y * y) / (v - c_y); 
If elevation of the camera from the ground is assumed to be known and called H, the equation becomes;
    z = (f_y * H*) / (v - c_y); 
Note that mean height of all vehicles in the KITTI dataset is 1.65m. Since the equation 3 is not continious and the result may be negative, the following depth scoring function is used;
    S_d = -ReLU(B * (v - c_y) / f_y * H) where B is a constant. 

#### 2.1.1.2 Semantic Scoring Function
For the semantic scoring function, a neural network is used to detect the keypoints from the images. A regression branch is added to the CenterNet. 
    S_s = f(H) where H is the input image feature obtained from the DLA-34 and f is the CNN architecture. 

Total score is calculated as;
    S = S_d + \alphaS_s where \alpha is a hyperparameter.

The loss of point detection is calculated as;
    L_CCE = FL(g^m(u_t, v_t), S) where FL is the Focal Loss, (ut , vt ) is the ground truth key point coordinate, g^m is the mapping function which turns m point coordinates
into heatmap.
After scoring is done for the whole feature map, mean value of each pixel score within a token is taken as the importance of the token. For the token clustering, a cluster center token is chosen which has the highest average score. In some stages, there may be more than one cluster center required. In that case, at each iteration, nth cluster center is chosen among the clusters which has the highest ranking.

### 2.1.2 Adaptive Token Transformer (ATT)
By using both the tokents from the initial stage (feature map) and selected cluster centers, tokens are grouped into clusters. Tokens in each cluster are merged and single token for every cluster is created. Then results are fed into a transformer. Adaptive token transformer exploits the long-range self-attention mechanism. 

Figure XX - ATT
As shown in the Figure XX, ATT loops N stages where at every stage first outline preferred token grouipng and then later, attention based feature merging are performed. 

#### 2.1.2.1 Outline-preferred Token Grouping
To avoid local feature correlation, clustering based on the spatial distance is not reasonable. A variation of nearest neighbor algorithm is used. The algorithm considers feature similarity along with the image distance while clustering. 

Eq 7 gelecek buraya.
The equation states that, for every point each cluster center is looped. Feature similarity and spatial distances are considered and balance between them is controlled by \Beta. Minimum value of a point against a cluster means that, the most appropriate value for the cluster is found.

#### 2.1.2.2 Attention-based Feature Merging
While merging tokens, instead of directly averaging the token features at every cluster, token features are averaged with the guidence of attention scores. 

Eq 8 gelecek paperdaki : where yi is the merged token feature, x_j and p_j are the original token features and they are summed over the i'th cluster set. 
Merged tokens are fed into the transformer as queries Q. Original tokens are also used as keys K and values V. Attention matrix of the transformer altered and attention score p is involved in the calculation in order to allow tokens to be more important. 

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + p)V where d_k is the number of the channels of the queries. 

Token Features might have dimensions (n,d), where n is the number of tokens and d is the feature dimension. Attention Scores might have a dimension of
(n,1), as each token gets a single score. To combine these (e.g., by addition), their shapes must match. If they don’t, matrix expansion ensures that smaller matrices (like (n,1)) are "stretched" to match larger ones (like (n,d)). Also, token attention score allows network to focus on the important information.

### 2.1.3 Multi-stage Feature Reconstruction (MFR)
Multi-stage Feature Reconstruction restores and aggregates all N stages. MFR upsamples the tokens from a history record and restores the feature map. In the Figure XX unsampling process is given.
FIGURE
In Attention-based Feature Merging every token is assigned to a cluster and since tokens are merged, every cluster is represented by a merged token. Positional correspondences between original and merged tokens are recorded and those records are used to copy the merged token features into the unsampled token. Token features from the previous stage are added iteratively to get an aggregate. Later, tokens are fed to a multilayer perceptron and that process is iterated N times. After the process a feature map of size (Ws x Hs x C') is obtained.

### 2.1.4 Monocular 3D Detection
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
