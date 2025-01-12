# MotionTrack: Learning Robust Short-term and Long-term Motions for Multi-Object Tracking

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

This project aims to introduce and implement the paper [MotionTrack: Learning Robust Short-term and Long-term Motions for Multi-Object Tracking](https://arxiv.org/pdf/2303.10404) [1], which was published in CVPR 2023. The goal of this project is to evaluate the reproducibility and the proposed solution in the paper. The paper introduces a new set of methods for object tracking, especially in crowded places.

Multi-Object Tracking (MOT) is one of the hot topics in the computer vision and machine learning domains. The primary challenge of MOT is determining accurate trajectories for objects across frames in videos, especially in dense crowds. Although some existing methods address this, handling object tracking in dense crowds (including lost objects) remains problematic. Hence, challenges such as the Multi Object Tracking (MOT) Challenge have been introduced over the past few years. This paper also uses MOT datasets to compare results with those from several other papers.

![figures/introduction.png](figures/introduction.png)

## 1.1. Paper summary

The paper focuses on solving problems in Multi-Object Tracking (MOT), such as keeping track of objects in crowded areas and when they are not visible to the camera. It introduces a method called MotionTrack, which combines two ideas:

- **Interaction Module**: Helps predict how objects will move in crowded spaces by understanding how they affect each other.
- **Refind Module**: Helps find and reconnect objects that were lost by using their past movements.

The paper shows that MotionTrack outperforms other methods on two popular datasets (MOT17 and MOT20). Its main contributions are:

- A way to predict movements in crowded areas.
- A way to reconnect lost objects by using their history.
- A method that works well with existing object detection systems.

More details can be found in Section 2.1.

# 2. The method and our interpretation

## 2.1. The original method

MotionTrack solves two main problems in tracking objects:

- Tracking objects between nearby frames (short-range association).
- Reconnecting objects that are lost for a while (long-range association).

It uses an approach called tracking-by-detection, where objects are first detected in each frame and then associated across the video.

![figures/general-overview.png](figures/general-overview.png)

### 2.1.1 Object Detection

In this paper, the frames are first processed by YOLOX [2] to detect the locations of objects. Then, the locations of the people detected in each frame, along with the delta values between frames, are used to train the algorithm.

### 2.1.2 Interaction Module (Short-range Association)

This module exists specifically to detect the relationships of people between frames. The main goal is to prevent losing track of people from frame to frame, while the Refind Module is responsible for rediscovering people who disappear.

- It creates a special matrix to detect how strongly one person affects another.
- A type of neural network is used with this matrix to predict how objects move.
- This makes it easier to track objects that might overlap or partially block each other in crowded scenes.

![figures/interaction-module.png](figures/interaction-module.png)

The figure above demonstrates how the interaction module works.

![figures/interaction1.png](figures/interaction1.png)

First, an attention mechanism is applied to the locations. `I` represents the input, namely `x, y, w, h, Δx, Δy, Δw, Δh`.

![figures/interaction2.png](figures/interaction2.png)

Next, the result of the attention is passed through two convolutional networks. The outputs of these networks are then added, and a PReLU function is applied.

![figures/interaction3.png](figures/interaction3.png)

Then, the result of the above operation is filtered by a sign function and undergoes an element-wise multiplication with the output of the attention matrix.

![figures/interaction4.png](figures/interaction4.png)

After that, the result of this operation is multiplied by the embedding of `O`, which represents the second part of the data input `I`. The results are then passed through a PReLU function.

Finally, these outputs are given to a Multi-Layer Perceptron (MLP). The final output, `Poffs`, represents the predicted delta values for `Δx, Δy, Δw, Δh` to be applied in the next frame.

### 2.1.3 Refind Module (Long-range Association)

This module specifically handles detecting or predicting the positions of lost people across frames. The algorithm uses past behaviors of the tracked people to rediscover them when they disappear.

- It uses an object’s past movements to calculate a score between the lost object and new detections.
- If a match is found, it fixes the object’s position during the time it was hidden to make tracking smoother.

The result of this module is fed back to the interaction module to continue creating associations between frames.

![figures/correlation-matrix.png](figures/correlation-matrix.png)

The figure above demonstrates how the correlation matrix works in the refind module.

![figures/refind1.png](figures/refind1.png)

The refind module takes two different sets of data as its input. The first set, `T`, includes the lost tracklets between the previous and current frames. The second set, `D`, represents the current non-matching (or newly matched) detections in the current frame. Each data point has 5 values: `t, x, y, w, h`. Here, `t` represents the frame ID.

First, the data `T` is passed through a convolutional network. The output is then passed through another convolutional network, followed by a pooling operation, resulting in `Ftraj`.

![figures/refind2.png](figures/refind2.png)

Next, `Drest` is calculated. `Drest` is the difference between each detection and the last known location of each tracklet. Then, a linear transformation is applied to get `Fdete`.

Finally, `Fdete` and `Ftraj` are combined into a future matrix, which is then passed through a fully connected layer and a sigmoid function to get the correlation matrix. This matrix represents the likelihood of matches between lost tracklets and detections.

### 2.1.5 Loss Functions

For the interaction module, the Intersection over Union (IoU) loss function is used. The exact formula is shown below:

![figures/interaction-loss.png](figures/interaction-loss.png)

For the refind module, a binary cross-entropy loss function is used. A 1-0 matrix is provided to compute this loss:

![figures/refind-loss.png](figures/refind-loss.png)

### 2.1.4 Overall Process

1. The system uses an object detection model (YOLOX [2]) to find objects in each frame.
2. The Interaction Module predicts where objects will move and associates them across frames.
3. If an object is missing, the Refind Module uses its past movements to find it and reconnect it.
4. The system combines all the information to create paths for each object.

## 2.2. Our interpretation

In the paper, it is not explicitly stated which trained version of YOLOX [2] is used. Through experimentation, We found that YOLOX trained with the `yolox_x` model yields the best results. We will use PyTorch to implement the Interaction and Refind modules, which are trained on the MOT17 and MOT20 datasets, as described in the paper.

The formats provided by the MOT datasets (MOT17 and MOT20) differ somewhat from what the paper requires. Therefore, We wrote a data conversion script to modify the dataset format before training. MOT provides the coordinates of detected people in each frame. However, the implementation requires both coordinates and size (width and height) for each detection.

We have set up an environment where the input is first passed to the interaction module. Based on the results of the interaction module, we predict the next position of each tracklet between frames. Next, the refind module is run in the pipeline to locate any lost tracklets.

The pipeline tracks each individual, including whether they are lost in a sequence of frames. Consequently, we must reset the pipeline’s memory for each dataset because they represent different sets of frames.

# 3. Experiments and results

## 3.1. Experimental setup

We used a MacBook Air M2 with 16 GB of RAM and an 8-core GPU to set up my environment. We will use PyTorch for the implementation. The inputs are first sent to YOLOX for object detection on a frame-by-frame basis. Then, the two-stage machine learning architecture (interaction and refind modules) is used to find the relationships between frames for each person.

## 3.2. Running the code

### 3.2.1 Setting up YOLOX and PyTorch

First, one has to setup YOLOX, pytorch and necessary modules to local. Python 3 is required to run the project.

1. Install Python 3.
2. Run `pip3 install -r requirements.txt` to install dependencies.
3. Run `git apply` for each patch in the `patches` folder.

We had to create some patches for the YOLOX library to ensure the compatibility of the code with our environment.

### 3.2.2 Downloading Pre-Trained YOLOX

Download a pre-trained version of YOLOX from [here](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/README.md).

### 3.2.3 Downloading Datasets

Download the [MOT17](https://motchallenge.net/data/MOT17/) and [MOT20](https://motchallenge.net/data/MOT20/) datasets. In these datasets, `gt.txt` files are used to train the system.

## 3.3. Results

In the paper, certain implementation details are not stated, such as:

- How many layers are used for the convolutional networks (denoted as `N`).
- The specifics of the MLP (Multi-Layer Perceptron).
- The exact way the correlation matrix is calculated from `Ftraj` and `Fdete`.
- How the metrics are computed.

Hence, we filled in these details with the most logical assumptions.

Below are some values we used, which differ from those in the paper:

|                                         | Paper             | Us        |
|-----------------------------------------|-------------------|-----------|
| Sigmoid Threshold in interaction module | 0.6               | 0.5       |
| Tracklet initialization score           | 0.7               | 0.5       |
| Refind module reject threshold          | 0.9               | 0.5       |
| Lost tracklet frame memory size         | 60 and 120 frames | 30 frames |

Below are the training losses of the interaction and refind modules.

![figures/train-loss-interaction.png](figures/train-loss-interaction.png)

![figures/train-loss-refind.png](figures/train-loss-refind.png)

# 4. Conclusion

In this project, we implemented the paper entitled “MotionTrack: Learning Robust Short-term and Long-term Motions for Multi-Object Tracking.” The paper introduces an algorithm to track people in crowded video scenes, particularly focusing on methods to rediscover lost individuals.

Although we have largely implemented the approach described in the paper, it is quite challenging to build a pipeline that is 100% compatible with the original specifications. First, it is difficult to connect the modules’ inputs and outputs while maintaining a memory of the tracklets and detections throughout training. Moreover, comparing the results with the original framework is challenging without completing both learning models.

The main idea provided by the paper is both simple and effective. Essentially, it introduces an algorithm to predict people’s behavior by understanding how they affect one another, using this information to reconnect lost individuals in crowded areas. By calculating a trajectory for each person, the paper links lost tracklets with detections.

# 5. References

1. Qin, Z., Zhou, S., Wang, L., Duan, J., Hua, G., & Tang, W. (2023). Motiontrack: Learning robust short-term and long-term motions for multi-object tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 17939-17948).
2. Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun., YOLOX: Exceeding yolo series in 2021. arXiv preprint, arXiv:2107.08430, 2021.

# Contact

OZAN AKIN [ozan.akin@metu.edu.tr](mailto:ozan.akin@metu.edu.tr)
