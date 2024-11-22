# Swift-Mapping: Online Neural Implicit Dense Mapping in Urban Scenes


# 1. Introduction

The paper, "Swift-Mapping: Online Neural Implicit Dense Mapping in Urban Scenes," [[1]](#1-wu-ke-kaizhao-zhang-mingzhe-gao-jieru-zhao-zhongxue-gan-and-wenchao-ding-swift-mapping-online-neural-implicit-dense-mapping-in-urban-scenes-proceedings-of-the-aaai-conference-on-artificial-intelligence-38-no-6-march-24-2024-604856-httpsdoiorg101609aaaiv38i628420) was presented at the Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24). It introduces an innovative framework for online dense mapping in urban environments using neural implicit representations. The proposed approach focuses on achieving high-fidelity and real-time 3D scene reconstruction while addressing computational challenges posed by dynamic and large-scale urban settings.

Our goal is to reproduce the results and gain a comprehensive understanding of the Swift-Mapping framework, including its Neural Implicit Octomap (NIO) and real-time mapping capabilities. This effort aims to ensure the reproducibility of the proposed methods and evaluate their applicability to broader domains such as robotics and autonomous navigation.

## 1.1. Paper summary

#### Context and Existing Literature

Traditional dense mapping approaches (e.g., SLAM, multi-view stereo) struggle in dynamic urban environments due to challenges like **occlusions**, **rapid motion**, and **scale variations**. Neural Radiance Fields (**NeRF**) offer high-fidelity reconstructions but are limited to **offline tasks** and require significant computational resources. Efforts like **iMAP** and **NICE-SLAM** bring neural implicit representations to online systems but are constrained to **indoor settings** or suffer from high memory demands.

#### Swift-Mapping’s Method

Swift-Mapping addresses these limitations with:
1. **Neural Implicit Octomap (NIO):**
   - A sparse octree-based voxel structure optimized for large-scale and dynamic urban scenes, using adaptive resolutions for efficient memory usage and accuracy.
2. **Online Dense Mapping Framework:**
   - Real-time updates through hybrid sampling, hierarchical latent features, and dynamic object modeling.
3. **Performance Innovations:**
   - Achieves **10x faster reconstruction** with **state-of-the-art accuracy** and robustness to challenges like **fast ego motion**.

#### Key Contributions

1. A scalable neural implicit mapping method designed for urban scenes.
2. Supports **dynamic obstacle modeling** and **scene editing** via feature voxel manipulation.
3. Demonstrates superior performance in speed and accuracy compared to both existing online and offline methods, enabling real-time applications for autonomous navigation.


# 2. The method and our interpretation

## 2.1. The original method

### Methodology Overview
The method introduced in this paper mainly fuses the RGB data from a camera and sparse  Point Cloud data from a LiDAR through a MLP Decoder to generate a map in the
form of octree structure. It firstly takes RGB and sparse Point Cloud data and utilizing the CompletionFormer [[2]](#2-y-zhang-x-guo-m-poggi-z-zhu-g-huang-and-s-mattoccia-completionformer-depth-completion-with-convolutions-and-vision-transformers-2023-ieeecvf-conference-on-computer-vision-and-pattern-recognition-cvpr-vancouver-bc-canada-2023-pp-18527-18536-doi-101109cvpr52729202301777) framework for depth completion to interpolate the sparse 
depth information to generate a dense depth inputs. Next, it initialize the octree structure to generate octomap using this information, sparsly sampling the depth information along each ray
and utilizing the the camera pose. After initializing the octomap, it generates the feature vectors representing the color and depth using trilinear interpolation. 
Meanwhile, it utilizes the positional encoding to generate the feature vector representing the position of each voxel along the map. The generated color and depth feature
vectors are combined with learnable memorization parameters to control the forgetting of the features between each frame. Generated feature vectors, including the 
position encoding, are then concatenated to generate a single feature vector and fed into a MLP decoder, whose architecture is selected as ConvOnet architecture, with 
5 fully-connected layers and a residual connection added to the 3rd layer. The network is trained with Photometric Loss, penalizing the loss of the RGB information, 
and Geometric Loss, penalizing the loss of the depth information, combined. Finally, the output of the MLP decoder is used to reconstruct the dense RGBD maps, or 3D meshes are generated 
using Marching Cubes. Moreover, MOTSFusion [[3]](#3-j-luiten-t-fischer-and-b-leibe-track-to-reconstruct-and-reconstruct-to-track-in-ieee-robotics-and-automation-letters-vol-5-no-2-pp-1803-1810-april-2020-doi-101109lra20202969183) framework is utilised to generate Moving Octree structure, modeling the dynamic environment without the need of retraining 
the network. Generated Moving Octree can also be utilised to modify the position of the objects offline. 

### Main Points
- *Neural Implicit Octomap (NIO):* 
  - Hierarchical octree structure for scalable and sparse representations.
  - Adaptive voxel resolution for fine-grained nearby mapping and coarse-grained distant mapping.
- *Fusion of RGB and LiDAR Data:*
  - RGB and sparse depth data are processed through *CompletionFormer* for dense depth completion.
- *Dynamic Scene Modeling:*
  - *MOTSFusion framework* creates a Moving Octree for modeling dynamic objects without retraining.
  - Scene editing functionality to modify object positions offline.

### Input Data
- *RGB Data:* Captured by a camera.
- *Sparse Point Cloud Data:* Captured by a LiDAR sensor.

### Steps of the Algorithm
1. *Depth Completion:*
   - Utilizes the CompletionFormer framework to interpolate sparse depth into dense depth inputs.

2. *Octree Initialization:*
   - Initializes the Neural Implicit Octomap (NIO) with hierarchical voxel grids.
   - Sparse sampling of depth information along rays using the camera pose.

3. *Distance Adaptive Voxel Initialization:*
   - Voxels are initialized with resolutions that adapt to the distance of the region from the camera.
   - Nearby regions use finer resolution, while distant regions use coarser resolution to optimize memory and computational efficiency.
   - Voxel resolution at level $` k `$ is calculated as:
   
   $$\text{Voxel Resolution at Level } k = l \cdot 2^k$$, 
   
   where $` l = \text{minimum resolution} `$.
   
   - Voxel resolutions are dynamically selected based on scale variation at different distances:
   
   $$2^{k-K} \cdot d_{\text{max}} \leq \text{distance} < 2^{k-K+1} \cdot d_{\text{max}},$$
   
   where $` d_{\text{max}} `$ is the maximum sampling distance.

4. *Feature Vector Generation:*
   - *Depth and Color Features:* Generated via trilinear interpolation from neighboring voxels.
   - *Position Encoding:* Encodes spatial position of voxels in the map.
   - *Trainable Memorization Parameters ($` \alpha_k `$ and $` \beta_k `$):* Control the forgetting and retention of features across frames.

   Each voxel at level $` k `$ is associated with latent feature vectors for depth ($` \phi_d^k `$) and color ($` \phi_c^k `$):

   $$\phi_d^k(p) = (\phi_d^k(p), \alpha_k \phi_d^k(p), \alpha_k^2 \phi_d^k(p), \dots, \alpha_k^{2^{K-k}} \phi_d^k(p))$$

   $$\phi_c^k(p) = (\phi_c^k(p), \beta_k \phi_c^k(p), \beta_k^2 \phi_c^k(p), \dots, \beta_k^{2^{K-k}} \phi_c^k(p)).$$

   These latent vectors are concatenated across all levels $` k `$ using a max-pooling operation:

   $$\phi_d(p) = \max_k (|\phi_d^k(p)|), \quad \phi_c(p) = \max_k (|\phi_c^k(p)|).$$

5. *MLP Decoding:*
   - Combines feature vectors and feeds them into an MLP decoder following the ConvOnet architecture.
   - The decoder comprises 5 fully connected layers with residual connections on the third layer.
   - Predictions for occupancy ($` o_p `$) and color ($` c_p `$) are made as follows:
   
   $$o_p = f_\theta^d(p, \phi_d(p)), \quad c_p = f_\omega^c(p, \phi_c(p)).$$

6. *Training Objectives:*
   - *Photometric Loss ($` L_p `$):* Penalizes RGB information loss.
   
   $$L_p = \frac{1}{M} \sum_{m=1}^M || I_m - \hat{I}_m ||^2$$

   - *Geometric Loss ($` L_d `$):* Penalizes depth information loss.
   
   $$L_d = \frac{1}{M} \sum_{m=1}^M || D_m - \hat{D}_m ||^2$$

   - Combined optimization target:
   
   $$\min_{\theta, \omega, \alpha, \beta, \phi_d, \phi_c} (\lambda_d L_d + \lambda_p L_p)$$

7. *Output Generation:*
   - Dense RGB-D maps.
   - 3D meshes generated using *Marching Cubes*.

8. *Dynamic Scene Handling:*
   - Utilizes MOTSFusion to create a Moving Octree, modeling dynamic obstacles without retraining.
   - Supports scene editing by manipulating voxel features offline.


### Main Steps
1. Feed RGB and sparse point cloud data.
2. Initialize the Neural Implicit Octomap with adaptive voxel resolutions.
3. Train the MLP decoder using photometric and geometric losses.
4. Generate dense RGB-D maps or 3D meshes.
5. Optionally, use the Moving Octree for dynamic scene modeling or offline editing.

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

#### [1] Wu, Ke, Kaizhao Zhang, Mingzhe Gao, Jieru Zhao, Zhongxue Gan, and Wenchao Ding. “Swift-Mapping: Online Neural Implicit Dense Mapping in Urban Scenes.” Proceedings of the AAAI Conference on Artificial Intelligence 38, no. 6 (March 24, 2024): 6048–56. https://doi.org/10.1609/aaai.v38i6.28420.
#### [2] Y. Zhang, X. Guo, M. Poggi, Z. Zhu, G. Huang and S. Mattoccia, "CompletionFormer: Depth Completion with Convolutions and Vision Transformers," 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, BC, Canada, 2023, pp. 18527-18536, doi: 10.1109/CVPR52729.2023.01777.
#### [3] J. Luiten, T. Fischer and B. Leibe, "Track to Reconstruct and Reconstruct to Track," in IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 1803-1810, April 2020, doi: 10.1109/LRA.2020.2969183.


# Contact

- **Yavuz Selim Özkaya**  
  Email: [e230518@metu.edu.tr](mailto:e230518@metu.edu.tr)  
  LinkedIn: [yavuzselimozkaya](https://www.linkedin.com/in/yavuzselimozkaya/)  
  GitHub: [ysozkaya](https://github.com/ysozkaya)

- **Eminalp Koyuncu**  
  Email: [e230788@metu.edu.tr](mailto:e230788@metu.edu.tr)  
  LinkedIn: [eminalp-koyuncu](https://www.linkedin.com/in/eminalp-koyuncu/)  
  GitHub: [eminalpkoyuncu](https://github.com/eminalpkoyuncu)
  