# Adaptive-SpikeNet: Event-based Optical Flow Estimation using Spiking Neural Networks with Learnable Neuronal Dynamics

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction
An event-based camera is a new sensor modality that operates by detecting changes in log pixel intensity rather than capturing entire frames. When a change in log intensity occurs at any pixel, the camera reports the pixel’s coordinates, the polarity of the change (indicating whether the intensity increased or decreased), and the precise timestamp of the event as a tuple {x,y,t,p} in Address Event Represenattion(AER). This unique approach enables the camera to achieve high temporal resolution, capturing rapid motion. It also delivers a high dynamic range, allowing it to perform effectively in scenes with extreme lighting contrasts. Furthermore, its design significantly reduces power consumption, making it an energy-efficient alternative to traditional frame-based imaging systems.

![MVSEC_indoor_fliying1](https://github.com/user-attachments/assets/b46e431d-e4d3-482d-9975-57121ccbdf1b)

![Events Camera](https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/output.gif)

The image above depicts a traditional image frame on the left, while the middle panel shows accumulated events over time. In the middle panel, blue represents positive polarity (increased intensity), and red represents negative polarity (decreased intensity). The rightmost panel, on the other hand, illustrates the optical flow.

The motion of each pixel between two successive images is referred to as dense optical flow. This information is crucial for numerous downstream tasks, including Simultaneous Localization and Mapping (SLAM) and odometry, where understanding motion dynamics is fundamental. While optical flow estimation from conventional image frames has been extensively studied and significantly advanced over the years, estimating optical flow from event-based cameras remains a challenging and less explored area due to the fundamentally different data representation and asynchronous nature of event streams.

A few attempts have been made to estimate optical flow from event streams using conventional neural networks, such as E-RAFT and EV-Flownet. These methods rely on accumulated events to compute the optical flow, effectively adapting traditional image-based approaches to event data. In contrast, Spike-FlowNet employs Integrate-and-Fire (IF) neurons, or spiking neurons, which are more aligned with the asynchronous and sparse nature of event streams.


## 1.1. Paper summary
### Proposed Framework
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

# Adaptive-SpikeNet: Event-Based Optical Flow Estimation Using Spiking Neural Networks with Learnable Neural Dynamics

## Overview

The **Adaptive-SpikeNet** architecture leverages the unique properties of event-based cameras and spiking neural networks (SNNs) to efficiently estimate optical flow from sparse and asynchronous event data. A key innovation of Adaptive-SpikeNet lies in its **input representation**, specifically designed to capture the unique **spatial** and **temporal** structure of event-based data more effectively.

## Event-Based Input Representation

Event cameras generate asynchronous streams of data, where each "event" corresponds to a change in brightness at a specific pixel, along with a timestamp. Unlike traditional frame-based cameras, this data is sparse and irregular. **Adaptive-SpikeNet** leverages these characteristics with the following features:

### Key Features of the Representation

- **Spatio-Temporal Encoding**:  
  Events are represented in a way that retains their precise spatial $(x, y)$ coordinates and temporal (time) occurrence. Instead of collapsing events into discrete frames, the representation incorporates continuous time information, allowing the network to model motion dynamics effectively.
  
- **Spike-Based Input Format**:  
  The input is converted into spike trains, which are native to spiking neural networks. The spike timing directly corresponds to the temporal sequence of events, enabling the network to process data in real-time.
  
- **Learnable Temporal Dynamics**:  
  Unlike conventional methods that use fixed temporal windows, Adaptive-SpikeNet learns the temporal relationships between events, enhancing the network's ability to capture motion patterns over varying time scales.
  
- **Event Binning or Aggregation**:  
  Events may be aggregated into temporal bins or voxel grids to balance spatial resolution and computational efficiency. This aggregation is done carefully to ensure that the spatial structure (neighboring pixels and edges) is preserved.
  
- **Preservation of Sparsity**:  
  The representation avoids introducing redundancy. Sparse event data remains sparse in the input to the network, maintaining energy efficiency and computational speed.

### Why This Matters

The combination of these techniques ensures that both spatial structure (e.g., object edges, motion direction) and temporal resolution (e.g., event timing, flow velocity) are preserved and utilized effectively. This tailored input representation enhances the network's ability to:

- Detect fine-grained motion patterns.
- Adapt to varying speeds of motion.
- Preserve critical information without overwhelming the computational pipeline.

In essence, the input representation in Adaptive-SpikeNet bridges the gap between the asynchronous, sparse nature of event camera data and the spiking neural network's processing capabilities.

## Event Binning Process

In the Adaptive-SpikeNet framework, event bins are used to structure the sparse, asynchronous event data into a format suitable for spiking neural networks. The binning process can be broken down into the following steps:

### 1. Event Camera Data Representation

Each event generated by an event camera is represented as:

$$
e_i = (x_i, y_i, t_i, p_i)
$$

Where:

- $x_i, y_i$: Spatial coordinates of the event on the sensor.
- $t_i$: Timestamp of the event.
- $p_i$: Polarity of the event (+1 for an increase in brightness, -1 for a decrease).

### 2. Temporal Binning

To process events in a structured manner, the continuous event stream is divided into temporal bins. Let the duration of the entire observation period be $T$, divided into $N$ equal intervals, each of duration $\Delta t$:

$$
\Delta t = \frac{T}{N}
$$

Each bin $B_k$ contains all events that occurred within the time interval:

$$
B_k = \{ e_i \mid t_k \leq t_i < t_{k+1} \}, \quad k = 1, 2, \dots, N
$$

Where $$ t_k = k \cdot \Delta t $$.

### 3. Spatial Aggregation:
Within each temporal bin, the events are aggregated spatially over the pixel grid to form a 2D representation. This can be done using a voxel grid representation or by creating a histogram of events at each pixel location:

$$
V_k(x, y) = \sum_{e_i \in B_k} p_i \cdot \delta(x - x_i, y - y_i)
$$

Where:

- $V_k(x, y)$ is the binned value at pixel $(x, y)$ in the $k$-th time bin.
- $\delta(x - x_i, y - y_i)$ is the Kronecker delta ensuring only events at $(x_i, y_i)$ contribute.

### 4. Input Volume for the Network:
The event bins across all time intervals are stacked to create a spatio-temporal voxel grid:

$$
V(x, y, k) = V_k(x, y), \quad k = 1, 2, \dots, N
$$

This 3D tensor $V$ (spatial dimensions $x$, $y$ and temporal dimension $k$) serves as the input to the spiking neural network.

### 5. Spike Encoding:
To feed this voxel grid into an SNN, the aggregated event values $V_k(x, y)$ are encoded as spike trains. A common approach is to use a threshold-based encoding, where a spike is generated if the value at a pixel exceeds a threshold:

$$
s(x, y, t) = 
\begin{cases} 
1 & \text{if } V(x, y, k) > \theta \\
0 & \text{otherwise}
\end{cases}
$$

Where $\theta$ is the threshold for spike generation, and $t$ corresponds to the time step within the bin.

### Advantages of Event Binning:
- **Temporal Resolution**: By dividing events into bins, temporal information is preserved, enabling the network to process motion over time.
- **Spatial Aggregation**: Events within each bin contribute to the spatial structure, highlighting edges and patterns.
- **Sparsity Preservation**: Only active bins (with significant event activity) contribute to the input, reducing computational overhead.


## Spike Encoding in Adaptive-SpikeNet

Spike encoding is the critical step that transforms the processed event bin data into input suitable for spiking neural networks (SNNs). Here's how the process works in detail:

#### 1. From Event Bins to Spikes
After constructing the voxel grid $V(x, y, k)$, the values at each pixel $(x, y)$ within each bin $k$ are converted into spikes using various encoding strategies. The primary goal is to map the intensity or activity information from the bin into spike timing or frequency, which SNNs use for computation.

**(a) Threshold-Based Encoding**:  
This is one of the simplest and most widely used methods. A spike is generated at a pixel if the bin's aggregated value exceeds a threshold:

$$
s_k(x, y) =
\begin{cases}
1 & \text{if } V(x, y, k) > \theta \\
0 & \text{otherwise}
\end{cases}
$$

Where:

- $s_k(x, y)$ represents the binary spike activity at pixel $(x, y)$ for the $k$-th time bin.
- $\theta$ is a user-defined or learned threshold.

This method ensures sparsity, as only a subset of pixels will spike in each bin.

**(b) Rate Coding**:  
Instead of a binary spike, the number of spikes generated at a pixel is proportional to the aggregated value in the bin:

$$
\text{Spike Rate}(x, y) \propto V(x, y, k)
$$

Rate coding may involve discretizing the aggregated value into a fixed number of spikes per bin.

**(c) Temporal Coding**:  
In temporal coding, the timing of the spike conveys the information. For example, a higher value in $V(x, y, k)$ might lead to an earlier spike in the bin's interval:

$$
t_{\text{spike}}(x, y) = t_k + \tau \cdot \left( 1 - \frac{V(x, y, k)}{\max(V)} \right)
$$

Where:

- $t_{\text{spike}}(x, y)$ is the spike time at pixel $(x, y)$.
- $\tau$ is the duration of the bin interval $\Delta t$.
- $\max(V)$ normalizes the bin values to a range $[0, 1]$.

This method is biologically plausible and retains temporal precision.

#### 2. Feeding Spikes into the SNN
Once the spikes are generated, they are passed to the SNN for processing. Spiking neurons in the network operate based on biologically-inspired models like the Leaky Integrate-and-Fire (LIF) or Adaptive LIF model, which processes spikes over time.

**Leaky Integrate-and-Fire Neuron**:  
A typical LIF neuron accumulates input spikes over time:

$$
\tau_m \frac{dV(t)}{dt} = -V(t) + I(t)
$$

Where:

- $V(t)$ is the neuron's membrane potential.
- $\tau_m$ is the membrane time constant.
- $I(t)$ is the synaptic input (spike activity).

When $V(t)$ crosses a threshold $V_{\text{thresh}}$, the neuron fires a spike, and the potential is reset:

$$
\text{if } V(t) \geq V_{\text{thresh}} \quad \text{then spike, reset } V(t) \to V_{\text{reset}}
$$

#### Adaptations in Adaptive-SpikeNet:
The paper introduces learnable dynamics for these neurons, enabling them to adjust their time constants $\tau_m$, thresholds $V_{\text{thresh}}$, and other parameters based on the input data. This adaptive mechanism improves the network's ability to capture complex motion patterns.

#### 3. Advantages of the Encoding Process
- **Sparse Computation**: By generating spikes only for active regions, the computational load is significantly reduced.
- **Retained Spatio-Temporal Information**: The precise timing and location of spikes ensure that both spatial and temporal dynamics are preserved.
- **Biological Plausibility**: Encoding strategies like temporal coding align closely with how biological neurons process information.
- **Flexibility with Dynamics**: The learnable parameters allow the system to adapt to different types of motion and input characteristics.

### Summary of the Workflow:
1. **Raw Event Data**: Captured as $(x, y, t, p)$.
2. **Temporal Binning**: Events grouped into bins $V(x, y, k)$.
3. **Spike Encoding**: Bin values converted into spike activity using threshold, rate, or temporal coding.
4. **SNN Processing**: Spikes input to spiking neurons with learnable dynamics to estimate optical flow.

Would you like to delve deeper into the learnable dynamics of the neurons or their training process?



## Learnable Neural Dynamics

Adaptive-SpikeNet also introduces learnable neural dynamics to adaptively learn the temporal patterns of events. The key aspect of this learning process is the use of **spiking neurons** that can respond to incoming events based on learned weights and synaptic dynamics.

### Spiking Neuron Model

Each spiking neuron $n$ in the network receives input from a set of event bins $B_k$ and produces output spikes. The state of the neuron can be represented as a voltage $V_n(t)$ that evolves over time:

$$
\tau \frac{dV_n(t)}{dt} = -V_n(t) + I_n(t)
$$

Where:

- $\tau$: Time constant of the neuron.
- $V_n(t)$: Membrane potential of the neuron at time $t$.
- $I_n(t)$: Input current to the neuron, which is a function of the incoming events and the synaptic weights.

When the membrane potential $V_n(t)$ exceeds a threshold $V_{th}$, the neuron emits a spike:

$$
\text{if } V_n(t) \geq V_{th}, \quad \text{spike occurs at time } t.
$$

### Training and Optimization

The network is trained to estimate optical flow by adjusting the synaptic weights using a supervised learning algorithm, typically based on backpropagation or a local learning rule that updates synapses based on the error signal derived from the network's output.

## Model Summary

In summary, **Adaptive-SpikeNet** consists of the following key components:

- Event-based input representation that encodes both spatial and temporal information.
- Event binning mechanism to structure asynchronous data into meaningful time intervals.
- Learnable spiking neurons that adaptively model the dynamics of event data.
- Optimization framework for estimating optical flow and fine-tuning the network parameters.

This architecture provides an efficient and biologically plausible way to process event-based data, making it ideal for real-time optical flow estimation in dynamic environments.

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
