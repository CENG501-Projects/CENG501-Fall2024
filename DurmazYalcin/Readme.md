# Adaptive-SpikeNet: Event-based Optical Flow Estimation using Spiking Neural Networks with Learnable Neuronal Dynamics

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction
An event-based camera is a new sensor modality that operates by detecting changes in log pixel intensity rather than capturing entire frames. When a change in log intensity occurs at any pixel, the camera reports the pixel’s coordinates, the polarity of the change (indicating whether the intensity increased or decreased), and the precise timestamp of the event as a tuple {x,y,t,p}. This unique approach enables the camera to achieve high temporal resolution, capturing rapid motion. It also delivers a high dynamic range, allowing it to perform effectively in scenes with extreme lighting contrasts. Furthermore, its design significantly reduces power consumption, making it an energy-efficient alternative to traditional frame-based imaging systems.

<div align="center">
  <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/output.gif" alt="description" width="80%">
</div>

The gifs above depicts a small interval from [MVSEC](https://daniilidis-group.github.io/mvsec/) dataset. 
- **Left**:A grayscale image captured by a conventional camera.
- **Middle**: Accumulated events, where blue indicates positive polarity and red indicates negative polarity.
- **Right**: A visualization of the optical flow, visualizing the motion vector for each pixel.

**Dense Optical Flow** refers to the motion of each pixel between two successive images. This information is critical for tasks like Simultaneous Localization and Mapping (SLAM) and odometry.

While optical flow estimation from conventional camera images has seen significant progress, estimating it from event-based cameras remains a challenging area. This is due to the unique, asynchronous nature of event streams and their fundamentally different data representation.

A few attempts have been made to estimate optical flow from event streams using conventional neural networks, such as [E-RAFT](https://github.com/uzh-rpg/E-RAFT) and [EV-FlowNet](https://github.com/daniilidis-group/EV-FlowNet). These approaches rely on accumulated events to adapt traditional image-based techniques for event data.

In contrast, Spike-FlowNet utilizes Integrate-and-Fire (IF) neurons, or spiking neurons. These are better suited to the asynchronous and sparse nature of event streams, providing a more natural fit.


## 1.1. Paper summary
[Adaptive Neural Spike Net](https://ieeexplore.ieee.org/document/10160551) introduces a U-Net architecture designed to estimate optical flow from event data. It incorporates a spiking neural network (SNN) to capture the high temporal resolution of events. The SNN is built using Integrate-and-Fire (IF) neurons, which will be explained in detail. The proposed framework is depicted in the illustration below.

<div align="center">
  <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/SpikeNetwork.png" alt="description" width="80%">
</div>

The figure above shows a hybrid model that combines artificial neural networks (ANNs) and spiking neural networks (SNNs). SNNs are used for the encoders, while the rest of the network relies on traditional ANNs.

The network takes event data as input, stored in a special format, and outputs optical flow at four different scales. Backpropagation is applied simultaneously across all scales.

If ground truth optical flow data is available in public datasets, we can use it to define a loss function and train the model directly. However, this is often not the case. To address this, the paper proposes a self-supervised training method that uses grayscale images to define a loss for the estimated optical flow. Both training approaches are discussed in detail. We will discuss both policiy in detail. 

# 2. The method and our interpretation

## 2.1. The original method
The original method has three main components:
- Spiking Neural Network (SNN)
- Input Representation of Event Stream
- Loss Function
We will examine each one seperatly.

### Spiking Neural Network
The original work begins by describing the Integrate-and-Fire (IF) neuron model.

<div align="center">
  <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/if_neuron.png" alt="description" width="500">
</div>

As shown in the figure above, a neuron produces an output (or spike) only when its accumulated state exceeds a set threshold. For this work, the authors use leaky Integrate-and-Fire (LIF) neurons. A network built with these neurons is called a spiking neural network (SNN)

As highlighted in the original paper, specialized hardware designed for spiking neural networks exists. On such hardware, spiking networks offer the advantage of significantly reduced power consumption, making them ideal for energy-efficient computations.

### Input Representation
One of the key challenges in working with event cameras is handling the input representation. Unlike traditional image frames, events are sparse and arrive asynchronously. Therefore, it becomes crucial to design a method to feed these events into a network while preserving their temporal resolution.

<div align="center">
  <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/InputFormat.png" alt="description" width="500">
</div>

The figure above, borrowed from [Spike-Net](https://arxiv.org/abs/2003.06696), illustrates how events are grouped into bins. To estimate the optical flow at a specific time instant $t_k$, the events within the time interval $(t_{k-1},t_{k+1})$ are divided into bins as shown in the figure. This binning process is applied separately for **on events** (positive polarity) and **off events** (negative polarity). Moreover, the bins are labeled as former group and latter group. As a result, we have 4 different categories for the bins. 

<div align="center">
  <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/bin_cat.png" alt="description" width="20%">
</div>

The proposed network takes a 4-channel event input. Each channel corresponds to one of the categories described above.


## Loss Function
Adaptive-SpikeNet employs two distinct loss paradigms—**supervised loss** and **self-supervised loss**—depending on the availability of labeled optical flow datasets. 

#### 1. Supervised Loss

This approach is used when ground truth optical flow labels are available for the dataset. The supervised loss directly compares the predicted optical flow $( \hat{u}, \hat{v} )$ to the ground truth flow $( u, v )$. The overall loss is calculated by summing the errors for all pixels.

$$
L_{\text{EPE}} = \frac{1}{N} \sum_{x, y} \left((\hat{u}(x,y) - u(x,y))^2 + (\hat{v}(x,y) - v(x,y))^2 \right)
$$

Where:

- $( \hat{u}(x, y), \hat{v}(x, y) )$ are the predicted horizontal and vertical flow components at pixel \( (x, y) \),
- $( u(x, y), v(x, y) )$ are the corresponding ground truth components,
- $( N )$ is the total number of pixels.


#### 2. Self-Supervised Loss

For datasets where ground truth optical flow is unavailable (e.g., real-world event-based datasets), Adaptive-SpikeNet employs gray-scale images to define a self-supervised loss. The loss has two components
- **Photometric Loss:**
  Assume that we have a predicted flow at time instant $t_k$. The photometric loss, $L_{\text{photo}}$ , inserts that when the grayscale image $I_k$ is warped using the estimated flow, it should be the same with the gray-scale image $I_{k+1}$.

$$
L_{\text{photo}} = \sum_{x, y} \rho\left(I_{k}(x + u, y + v) - I_{k+1}(x, y)\right)
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; where $\rho(\cdot)$ is the Charbonnier loss. 
  
- **Smoothness Loss:**
  On the other hand, the smoothness loss, $L_{\text{smooth}}$​, enforces the assumption that neighboring pixels should exhibit similar optical flow values. While this may not hold true for every neighboring pixel, it is a reasonable assumption for most of the pixels within the image. To balance its contribution to the overall loss.

$$
L_{\text{smooth}} = \sum_{j}\sum_{i} \left( \lVert u_{i,j} - u_{i+1,j} \rVert + \lVert u_{i,j} - u_{i,j+1} \rVert + \lVert v_{i,j} - v_{i+1,j} \rVert + \lVert v_{i,j} - v_{i,j+1} \rVert \right)
$$


Finally, the overall loss is expressed as

$$
L^{u} = L_{\text{photo}} + \alpha L_{\text{smooth}}
$$

where $\alpha$ is a hyperparameter to scale the effect of smoothness loss. 

## 2.2. Our interpretation
The network employs a straightforward U-Net architecture, which has many similar examples available online. However, the following considerations are crucial:
- The input representation should be manipulated carefully.
- Unlike conventional ANNs, PyTorch does not natively support SNN structures. For this purpose, the [Spike-FlowNet GitHub](https://github.com/chan8972/Spike-FlowNet)  repository offers an implementation of SNNs, which can be adapted for this work.
- Special attention must be paid to implementation of the loss function, as it directly impacts the network's performance for estimating the optical flow. 

### Input Representation and Scaled Flows
#### Visualization of Input Fromat
Assume that we desire to estimate the optical flow at time instant $t_k$. We determine the half window size as $100$ msec. Then, we create the bins between time interval $(t_k-100,t_k+100)$ and visualize them in order to verify the implementation.

#### Scaled Flows
When backpropagation is applied at four different scales of optical flow, we adjust the motion values (optical flow) for each scale. For example, if an image is resized to be twice as large, the motion of each pixel also doubles. To ensure the motion stays accurate, we scale the optical flow values to match the size of the image at each scale. This way, the optical flow at each scale correctly represents the motion for that specific scale.

```python
def get_scaled_flows(flow:np.array):
    scaled_flows = []
    old_height = flow.shape[0]
    old_width  = flow.shape[1]

    # Resize the flows by (1,2,4,8)
    for idx in range(4):
        new_height = int(old_height / (2**idx))
        new_width  = int(old_width / (2**idx))
        # Resize and divide by (2**idx)
        scaled_flow = cv2.resize(flow, (new_width,new_height), interpolation=cv2.INTER_LINEAR) / (2**idx)
        scaled_flows.append(scaled_flow)
    return scaled_flows
```

### Integrate and Fire Neurons
We utilize the IF neuron implementation from the [Spike-FlowNet GitHub](https://github.com/chan8972/Spike-FlowNet) as a starting point. However, the provided IF neuron does not include leakage. Since the proposed method requires a leaky IF neuron, we incorporate the leakage mechanism into the implementation.
```python
# IF neuron from Spike-FlowNet
def IF_Neuron(membrane_potential, threshold):
    global threshold_k
    threshold_k = threshold
    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold_k, 0)
    membrane_potential = membrane_potential - ex_membrane # hard reset
    # generate spike
    out = SpikingNN()(ex_membrane)
    out = out.detach() + (1/threshold)*out - (1/threshold)*out.detach()

    return membrane_potential, out
```
### Loss Function
To implement supervised training, we need to incorporate the ground truth optical flow. However, as you may have noticed in the provided ground truth samples, not all pixels in a frame have corresponding ground truth optical flow. Therefore, we must first mask the pixels without valid ground truth values. The mask can be defined as follows:

$$
M(x, y) =
\begin{cases} 
1 & \text{if } \text{groundtruth}(x, y) \text{ is available}, \\
0 & \text{otherwise}.
\end{cases}
$$

The loss function is adjusted accordingly:

$$
L_{\text{EPE}} = \frac{1}{N} \sum_{x, y} M(x, y) \left((\hat{u}(x,y) - u(x,y))^2 + (\hat{v}(x,y) - v(x,y))^2 \right)
$$


# 3. Experiments and results

## 3.1. Experimental setup
For the primal tests, we use the [DSEC](https://dsec.ifi.uzh.ch/) dataset because it provides highly accurate groundtruth data. However, not all sequences in the [DSEC](https://dsec.ifi.uzh.ch/) dataset include groundtruth optical flow. Therefore, we download and utilize only the sequences that contain groundtruth. Please organize the folder structure as follows:
```
- path_to_dataset
  - thun_00_a
  - zurich_city_01_a
  - zurich_city_02_a
    zurich_city_02_a_optical_flow_forward_timestamps.txt
    - zurich_city_02_a_events_left
      events.h5
      rectify_map.h5
    - zurich_city_02_a_optical_flow_forward_event
      <flow_name_0>.png
      <flow_name_1>.png
      <flow_name_1>.png
      ...
  - zurich_city_02_c
  - zurich_city_02_d
  ...
```

During training, it is necessary to create the event bins for each optical flow groundtruth. However, this binning process is time-consuming and is repeated for every training epoch. To reduce the training time, pre-binning the relevant events and saving them in a separate folder can significantly reduce training time. By doing this, we can directly load the relevant events from the pre-binned files, eliminating the need for repeated binning during each epoch. 

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
