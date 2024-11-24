# Adaptive-SpikeNet: Event-based Optical Flow Estimation using Spiking Neural Networks with Learnable Neuronal Dynamics

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction
An event-based camera is a new sensor modality that operates by detecting changes in pixel intensity rather than capturing entire frames. When a change in intensity occurs at any pixel, the camera reports the pixel’s coordinates, the polarity of the change (indicating whether the intensity increased or decreased), and the precise timestamp of the event. This unique approach enables the camera to achieve high temporal resolution, capturing rapid motion. It also delivers a high dynamic range, allowing it to perform effectively in scenes with extreme lighting contrasts. Furthermore, its design significantly reduces power consumption, making it an energy-efficient alternative to traditional frame-based imaging systems.

![MVSEC_indoor_fliying1](https://github.com/user-attachments/assets/b46e431d-e4d3-482d-9975-57121ccbdf1b)

![Events Camera](https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/output.gif)

The image above depicts a traditional image frame on the left, while the middle panel shows accumulated events over time. In the middle panel, blue represents positive polarity (increased intensity), and red represents negative polarity (decreased intensity). The rightmost panel, on the other hand, illustrates the optical flow.

The motion of each pixel between two successive images is referred to as dense optical flow. This information is crucial for numerous downstream tasks, including Simultaneous Localization and Mapping (SLAM) and odometry, where understanding motion dynamics is fundamental. While optical flow estimation from conventional image frames has been extensively studied and significantly advanced over the years, estimating optical flow from event-based cameras remains a challenging and less explored area due to the fundamentally different data representation and asynchronous nature of event streams.

A few attempts have been made to estimate optical flow from event streams using conventional neural networks, such as E-RAFT and EV-Flownet. These methods rely on accumulated events to compute the optical flow, effectively adapting traditional image-based approaches to event data. In contrast, Spike-FlowNet employs Integrate-and-Fire (IF) neurons, or spiking neurons, which are more aligned with the asynchronous and sparse nature of event streams.


## 1.1. Paper summary
### Porposed Framework
The proposed method utilizes a simple U-Net framework to estimate the optical flow as illustrated below.
![Network](https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/SpikeNetwork.png)

Method (a) represents a framework from prior work that is already publicly available. In contrast, method (b) refers to the newly proposed framework, which is not yet publicly accessible. Our implementation will focus on method (b).

### Loss Function
Loss function, $L^{u}$, is defined through two constraints

$$
L^{u} = l_{\text{photo}} + \alpha l_{\text{smooth}}
$$

The photometric loss, $l_{\text{photo}}$, inserts that when the grayscale image frame at time $t+\Delta t$ is warped backward to align with the image frame at time $t$ using the estimated optical flow, the two frames should appear identical. Any discrepancy or inconsistency between the frames highlights a potential error in the estimated optical flow vectors.

$$
l_{\text{photo}} = \sum_{x, y} \rho\left(I_t(x, y) - I_{t + \Delta t}(x + u, y + v)\right)
$$

On the other hand, the smoothness loss, $l_{\text{smooth}}$​, enforces the assumption that neighboring pixels should exhibit similar optical flow values. While this may not hold true for every neighboring pixel, it is a reasonable assumption for most pixels within the image. To balance its contribution to the overall loss, this term is scaled by a parameter $\alpha$.

$$
l_{\text{smooth}} = \sum_{j}\sum_{i} \left( \lVert u_{i,j} - u_{i+1,j} \rVert + \lVert u_{i,j} - u_{i,j+1} \rVert + \lVert v_{i,j} - v_{i+1,j} \rVert + \lVert v_{i,j} - v_{i,j+1} \rVert \right)
$$

# 2. The method and our interpretation

## 2.1. The original method

The original work begins by describing what a spiking neural network (SNN) is. For the convenience of the reader, we also provide a brief explanation of the Integrate-and-Fire (IF) neuron model.

<img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/if_neuron.png" alt="description" width="600">

As illustrated in the figure above, a neuron generates an output (or spike) only when its integrated state surpasses a predefined threshold. Due to the nature of our problem, the authors of the paper employ leaky Integrate-and-Fire (LIF) neurons. A neural network constructed using IF neurons is referred to as a spiking neural network.

As highlighted in the original paper, specialized hardware designed for spiking neural networks exists. On such hardware, spiking networks offer the advantage of significantly reduced power consumption, making them ideal for energy-efficient computations.

The original method presents a hybrid model integrating conventional neural networks (CNNs) with spiking neural networks (SNNs). While SNNs are specifically employed for designing the encoders, the rest of the network operates on conventional image frames. Although SNNs do not enhance the system's performance, they significantly reduce inference time and energy consumption when deployed on appropriate hardware.

## 2.2. Our interpretation
@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

* We can say that we have already dowloaded and examined the event datasets. We can refer to the visualization and our visualization codes. 
* We can explain input representation.
* We can explain how to back propagate the photometric loss? How to compute the jacobian of the photometric loss?

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
