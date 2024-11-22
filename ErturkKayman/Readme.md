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

### 2.1.3 Multi-stage Feature Reconstruction (MFR)

### 2.1.4 Monocular 3D Detection


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
