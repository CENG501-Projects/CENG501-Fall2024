# Adaptive-SpikeNet: Event-based Optical Flow Estimation using Spiking Neural Networks with Learnable Neuronal Dynamics

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction
An event-based camera is a new sensor modality that operates by detecting changes in log pixel intensity rather than capturing entire frames. When a change in log intensity occurs at any pixel, the camera reports the pixel’s coordinates, the polarity of the change (indicating whether the intensity increased or decreased), and the precise timestamp of the event as a tuple {x,y,t,p} in Address Event Represenattion(AER). This unique approach enables the camera to achieve high temporal resolution, capturing rapid motion. It also delivers a high dynamic range, allowing it to perform effectively in scenes with extreme lighting contrasts. Furthermore, its design significantly reduces power consumption, making it an energy-efficient alternative to traditional frame-based imaging systems.


<p float="left">
  <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/MVSEC_indoor_flying1.gif" width="100%" />
  <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/output.gif" width="100%" />
</p>

The gifs above depicts traditional image frames on the left, while the middle panels shows accumulated events over time, MVSEC on the top, and the DSEC at the bottom. In the middle panel, blue represents positive polarity (increased intensity), and red represents negative polarity (decreased intensity). The rightmost panel, on the other hand, illustrates the optical flow.

The motion of each pixel between two successive images is referred to as dense optical flow. This information is crucial for numerous downstream tasks, including Simultaneous Localization and Mapping (SLAM) and odometry, where understanding motion dynamics is fundamental. While optical flow estimation from conventional image frames has been extensively studied and significantly advanced over the years, estimating optical flow from event-based cameras remains a challenging and less explored area due to the fundamentally different data representation and asynchronous nature of event streams.

A few attempts have been made to estimate optical flow from event streams using conventional neural networks, such as E-RAFT and EV-Flownet. These methods rely on accumulated events to compute the optical flow, effectively adapting traditional image-based approaches to event data. In contrast, Spike-FlowNet employs Integrate-and-Fire (IF) neurons, or spiking neurons, which are more aligned with the asynchronous and sparse nature of event streams.


## 1.1. Paper summary
### Proposed Framework
The proposed method utilizes a simple U-Net framework to estimate the optical flow as illustrated below.
![Network](https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/SpikeNetwork.png)

Method (a) represents a framework from prior work that is already publicly available. In contrast, method (b) refers to the newly proposed framework, which is not yet publicly accessible. Our implementation will focus on method (b).
## Loss Function
Adaptive-SpikeNet employs two distinct loss paradigms—**supervised loss** and **self-supervised loss**—depending on the availability of labeled optical flow datasets. Below is a detailed explanation of how these approaches are utilized:

## 1. Supervised Loss

This approach is used when ground truth optical flow labels are available for the dataset, such as in datasets specifically created for optical flow tasks.

### Loss Definition:
The supervised loss directly compares the predicted optical flow $( \hat{f}(x, y) )$ to the ground truth flow $( f_{\text{true}}(x, y) )$.

#### (a) End-Point Error (EPE):
The End-Point Error (EPE) is the primary supervised loss function:

$$
L_{\text{EPE}} = \frac{1}{N} \sum_{(x, y)} \left( (\hat{u}(x, y) - u(x, y))^2 + (\hat{v}(x, y) - v(x, y))^2 \right)
$$

Where:

- $( \hat{u}(x, y), \hat{v}(x, y) )$ are the predicted horizontal and vertical flow components at pixel \( (x, y) \),
- $( u(x, y), v(x, y) )$ are the corresponding ground truth components,
- $( N )$ is the total number of pixels.


## 2. Self-Supervised Loss

For datasets where ground truth optical flow is unavailable (e.g., real-world event-based datasets), Adaptive-SpikeNet employs a self-supervised loss based on the concept of **photometric consistency**.

### Loss Definition:
The self-supervised loss assumes that pixel intensities remain consistent across consecutive frames, except for motion-induced changes. It uses warping techniques to estimate the consistency of pixel intensities.

#### (a) Photometric Loss:
The photometric loss, $l_{\text{photo}}$, inserts that when the grayscale image frame at time $t+\Delta t$ is warped backward to align with the image frame at time $t$ using the estimated optical flow, the two frames should appear identical. Any discrepancy or inconsistency between the frames highlights a potential error in the estimated optical flow vectors.

$$
l_{\text{photo}} = \sum_{x, y} \rho\left(I_t(x, y) - I_{t + \Delta t}(x + u, y + v)\right)
$$

#### (b) Smoothness Loss:
On the other hand, the smoothness loss, $l_{\text{smooth}}$​, enforces the assumption that neighboring pixels should exhibit similar optical flow values. While this may not hold true for every neighboring pixel, it is a reasonable assumption for most pixels within the image. To balance its contribution to the overall loss, this term is scaled by a parameter $\alpha$.

$$
l_{\text{smooth}} = \sum_{j}\sum_{i} \left( \lVert u_{i,j} - u_{i+1,j} \rVert + \lVert u_{i,j} - u_{i,j+1} \rVert + \lVert v_{i,j} - v_{i+1,j} \rVert + \lVert v_{i,j} - v_{i,j+1} \rVert \right)
$$

### Combined Loss:
The self-supervised loss combines the photometric and smoothness terms:

$$
L^{u} = l_{\text{photo}} + \alpha l_{\text{smooth}}
$$

# 2. The method and our interpretation

## 2.1. The original method

The original work begins by describing what a spiking neural network (SNN) is. For the convenience of the reader, we also provide a brief explanation of the Integrate-and-Fire (IF) neuron model.

<img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/if_neuron.png" alt="description" width="600">

As illustrated in the figure above, a neuron generates an output (or spike) only when its integrated state surpasses a predefined threshold. Due to the nature of our problem, the authors of the paper employ leaky Integrate-and-Fire (LIF) neurons. A neural network constructed using IF neurons is referred to as a spiking neural network.

As highlighted in the original paper, specialized hardware designed for spiking neural networks exists. On such hardware, spiking networks offer the advantage of significantly reduced power consumption, making them ideal for energy-efficient computations.

The original method presents a hybrid model integrating conventional neural networks (CNNs) with spiking neural networks (SNNs). While SNNs are specifically employed for designing the encoders, the rest of the network operates on conventional image frames. Although SNNs do not enhance the system's performance, they significantly reduce inference time and energy consumption when deployed on appropriate hardware.

## 2.2. Our interpretation

### Overview

The **Adaptive-SpikeNet** architecture leverages the unique properties of event-based cameras and spiking neural networks (SNNs) to efficiently estimate optical flow from sparse and asynchronous event data. A key innovation of Adaptive-SpikeNet lies in its **input representation**, specifically designed to capture the unique **spatial** and **temporal** structure of event-based data more effectively.

### Event-Based Input Representation

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

### Event Binning Process

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

Where $ t_k = k \cdot \Delta t $.

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

**Synaptic Input and Spike Timing in Adaptive-SpikeNet**


In a **Spiking Neural Network (SNN)**, the synaptic input at any given time is influenced by the spikes arriving from other neurons or pixels. The **synaptic input** at a specific time depends on both **spatial** and **temporal** aspects of the incoming spikes. This mechanism is particularly relevant in **event-based** systems such as the **Adaptive-SpikeNet** architecture for optical flow estimation.

**1. Synaptic Input 𝐼(𝑡) Calculation**

The synaptic input 𝐼(𝑡) for a given neuron at time 𝑡 is determined by the spikes that have occurred before that time, as well as their spatial locations. The general formula for computing the synaptic input is:

-Mathematical Formulation

For a pixel \((x, y)\), the synaptic input at time \(t\) is the sum of the contributions from all spikes that occurred before \(t\):

$$
I(t) = \sum_{(x', y')}\sum_{t_{spike}(x', y') < t} w(x', y') \cdot e(t - t_{spike}(x', y'))
$$

Where:

- $( I(t) )$ is the synaptic input at time $( t )$,
- $( w(x', y') )$ is the synaptic weight from pixel $((x', y'))$ (a learned parameter),
- $( t_{spike}(x', y') )$ is the timestamp of a spike from pixel $((x', y'))$,
- $( e(t - t_{spike}(x', y')) )$ is the event-based function modeling the contribution of a spike from pixel $((x', y'))$ at time $( t_{spike}(x', y') )$.

- Event Function

The event function $( e(t - t_{spike}) )$ represents the temporal influence of a spike over time. It is often modeled as a **decaying function**, such as exponential decay or a Gaussian kernel. For example:

$$
e(t - t_{spike}) = \exp\left( -\frac{(t - t_{spike})^2}{2\sigma^2} \right)
$$

Where:

- $( \sigma )$ controls the spread of the temporal influence of each spike.

**2. Spike Activity and Temporal Dynamics**

The timing of the spikes is crucial in determining the synaptic input:

- **Earlier spikes**: Have a stronger impact on the synaptic input at later times, depending on the decay kernel.
- **Recent spikes**: Have a more significant effect on the synaptic input at time \( t \), but their influence decays over time.

This mechanism allows the SNN to process both **spatial** and **temporal** dependencies from the event-based data.

**3. Temporal and Spatial Dependencies in Adaptive-SpikeNet**

In the context of **Adaptive-SpikeNet** for **optical flow estimation**, the synaptic input mechanism is used to capture both the **temporal dynamics** and **spatial dependencies** of event-based data.

- **Motion Estimation**: The synaptic input at each neuron corresponds to the spikes from neighboring pixels, which encode changes in the scene.
- **Optical Flow**: By processing the spikes over time, the network learns to associate temporal spike patterns with the motion of objects in the scene. The optical flow $( u(x, y) )$ is estimated by examining how the event-based data evolves over time.

**4. Summary of Synaptic Input Mechanism**

- The synaptic input $( I(t) )$ is calculated by summing the weighted contributions of all past spikes, where the weights and the decay kernel control the influence of each spike.
- The **temporal dynamics** of spikes, combined with the **spatial arrangement** of pixels, are essential for capturing motion in event-based systems.
- Adaptive-SpikeNet uses these mechanisms to **learn the temporal patterns** of spikes and estimate motion (optical flow) in real-time.



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



### Training and Optimization Event-Based Optical Flow Estimation Example

To better comprehend the algorithm, let us give an example and explain the procedure on this example. This example demonstrates how event data is collected, and converted into a usable spike format, and how loss is calculated. We work through a 3×3 pixel frame over 3 timestamps. The steps are explained systematically.

## Step-by-Step Procedure

### Step 1: Simulate Event Data

We begin by simulating event data for a 3×3 pixel array. Each pixel records events with the following format:

- **(x, y)**: Pixel location
- **t**: Timestamp of the event
- **p**: Polarity of the event (1 for intensity increase, -1 for intensity decrease)

The following table shows a hypothetical event stream over 3 timestamps:

| Pixel (x, y) | Timestamp (t) | Polarity (p) |
|--------------|---------------|--------------|
| (0, 0)       | t₁ = 0.1      | +1           |
| (1, 1)       | t₁ = 0.1      | -1           |
| (2, 2)       | t₂ = 0.2      | +1           |
| (1, 0)       | t₂ = 0.2      | +1           |
| (0, 2)       | t₃ = 0.3      | -1           |
| (2, 1)       | t₃ = 0.3      | +1           |

### Step 2: Aggregate Events into Temporal Bins

Next, we divide the time range into bins to create event frames. Assume each bin represents a 0.1-second interval. The data is binned as follows:

- **Bin 1** (t = [0, 0.1]):
    - Pixel (0, 0): +1
    - Pixel (1, 1): -1

- **Bin 2** (t = [0.1, 0.2]):
    - Pixel (2, 2): +1
    - Pixel (1, 0): +1

- **Bin 3** (t = [0.2, 0.3]):
    - Pixel (0, 2): -1
    - Pixel (2, 1): +1

### Step 3: Convert Event Frames into Spikes

We use rate coding to convert the event frames into spikes. Each event corresponds to one spike over the time interval. The polarity of the event determines whether the spike is excitatory or inhibitory:

- Positive polarity (+1) corresponds to an excitatory spike.
- Negative polarity (-1) corresponds to an inhibitory spike.

For each bin, we convert the event data into spike frames.

### Step 4: Predict Optical Flow

Once we have the spike frames, we predict the optical flow (**f̂**) at each pixel using a spiking neural network. Assume the predicted optical flow for each pixel is as follows:

| Pixel (x, y) | Predicted Flow (û, v̂) | Ground Truth Flow (u, v) |
|--------------|-------------------------|--------------------------|
| (0, 0)       | (0.2, 0.1)              | (0.3, 0.1)               |
| (1, 1)       | (0.0, -0.2)             | (0.0, -0.3)              |
| (2, 2)       | (0.1, 0.3)              | (0.1, 0.3)               |

### Step 5: Calculate Loss

#### (a) End-Point Error (EPE) Loss

The **End-Point Error (EPE)** is calculated for each pixel using the following formula:

$$
EPE(x, y) = (û - u)^2 + (v̂ - v)^2
$$

For our example:

- **Pixel (0, 0)**: EPE = \((0.2 - 0.3)^2 + (0.1 - 0.1)^2 = 0.1\)
- **Pixel (1, 1)**: EPE = \((0.0 - 0.0)^2 + (-0.2 - (-0.3))^2 = 0.1\)
- **Pixel (2, 2)**: EPE = \((0.1 - 0.1)^2 + (0.3 - 0.3)^2 = 0.0\)

The total EPE loss is the mean of the individual pixel errors:

$$
L_{EPE} = \frac{0.1 + 0.1 + 0.0}{3} = 0.0667
$$

#### (b) Photometric Consistency Loss

The **photometric consistency loss** compares the warped event frame at \( t_1 \) using the predicted flow with the event frame at \( t_2 \). The formula is:

$$
L_{photometric} = \frac{1}{N} \sum |I_{t1}(x, y) - I_{t2}(x + û, y + v̂)|
$$

##### Step 1: Warp Event Frame Using Predicted Flow

For simplicity, we warp Bin 1 to align with Bin 2 using the predicted flow. The predicted optical flow for each pixel is:

| Pixel (x, y) | Predicted Flow (û, v̂) |
|--------------|-------------------------|
| (0, 0)       | (0.2, 0.1)              |
| (1, 1)       | (0.0, -0.2)             |
| (2, 2)       | (0.1, 0.3)              |

##### Step 2: Calculate Photometric Loss

The warped frame and target frame are compared pixel by pixel. For the nonzero pixels:

- At **(0, 0)**: \(|0 - 0| = 0\)
- At **(1, 1)**: \(|-1 - 0| = 1\)
- At **(2, 2)**: \(|+1 - +1| = 0\)

The total photometric loss is:

$$
L_{photometric} = \frac{0 + 1 + 0}{3} = 0.333
$$

#### (c) Smoothness Loss

**Smoothness loss** penalizes large gradients in the flow, encouraging smooth optical flow fields across neighboring pixels. The formula is:

$$
L_{smoothness} = \frac{1}{N} \sum \left( | \nabla û(x, y) | + | \nabla v̂(x, y) | \right)
$$

Where \( \nabla û \) and \( \nabla v̂ \) are the gradients of the predicted optical flow in the \( x \) and \( y \) directions, respectively.

## Step 1: Warp Event Frame Using Predicted Flow

In this step, we warp one event frame (Bin 1) to another (Bin 2) using the predicted optical flow. The process involves the following:

### 1.1 Predicted Optical Flow for Each Pixel

For each pixel in the event frame, the predicted optical flow is provided as $(\hat{u}, \hat{v})$, where $\hat{u}$ is the horizontal displacement and $\hat{v}$ is the vertical displacement:

| Pixel (x, y) | Predicted Flow $(\hat{u}, \hat{v})$ |
|--------------|-------------------------|
| (0, 0)       | (0.2, 0.1)              |
| (1, 1)       | (0.0, -0.2)             |
| (2, 2)       | (0.1, 0.3)              |

### 1.2 Event Frames for Bin 1 and Bin 2

- **Bin 1 (t = [0, 0.1])**:
[ +1 0 0 0 -1 0 0 0 0 ]

- **Bin 2 (t = [0.1, 0.2])**:
[ 0 0 0 +1 0 0 0 0 +1 ]



### 1.3 Warping Bin 1

The new position (x', y') for each pixel after warping is calculated using the formula:

$$
(x', y') = (x + \hat{u}, y + \hat{v})
$$

For example:

- Pixel (0, 0) with predicted flow (0.2, 0.1) gives a new position of (0.2, 0.1), which rounds to (0, 0).
- Pixel (1, 1) with predicted flow (0.0, -0.2) gives a new position of (1.0, 0.8), which rounds to (1, 1).
- Pixel (2, 2) with predicted flow (0.1, 0.3) gives a new position of (2.1, 2.3), which rounds to (2, 2).

Thus, the **Warped Bin 1** frame becomes:
[ 0 -1 0 0 0 0 0 0 +1 ]


## Step 2: Calculate Photometric Loss

The photometric loss is calculated as the difference between the warped frame $I_{t1}$ and the target frame $I_{t2}$. It is given by the formula:

$$
L_{\text{photo}} = \frac{1}{N} \sum_{x, y} |I_{t1}(x', y') - I_{t2}(x, y)|
$$

Where $N$ is the total number of pixels in the image, and $( I_{t1}(x', y') )$ and $( I_{t2}(x, y) )$ represent the pixel values in the warped frame and the target frame, respectively.

For the nonzero pixels:

- At $(0, 0): |0 - 0| = 0$
- At $(1, 1): |-1 - 0| = 1$
- At $(2, 2): |+1 - +1| = 0$

Thus, the **Total Photometric Loss** is:

$$
L_{\text{photo}} = \frac{1}{3} \times (0 + 1 + 0) = 0.333
$$

## Step 3: Calculate Smoothness Loss

Smoothness loss encourages smooth optical flow fields across neighboring pixels and is given by the formula:

$$
L_{\text{smooth}} = \frac{1}{N} \sum_{x, y} \left( \| \nabla \hat{u} (x, y) \| + \| \nabla \hat{v} (x, y) \| \right)
$$

Where $∇\hat{u}(x, y)$ and $∇\hat{v}(x, y)$ are the gradients of the optical flow components $\hat{u}$ and $\hat{v}$, calculated with respect to neighboring pixels.

The gradients for $\hat{u}$ and $\hat{v}$ are calculated using finite differences:

### 3.1 Gradients for $\hat{u}$

The gradient for $\hat{u}$ at each pixel $(x, y)$ is calculated as:

$$
\nabla \hat{u} (x, y) = |\hat{u} (x+1, y) - \hat{u}(x, y)| + |\hat{u} (x, y+1) - \hat{u} (x, y)|
$$

For example:

- At $(0, 0): |0.0 - 0.2| + |0.0 - 0.2| = 0.4$
- At $(1, 1): |0.0 - 0.0| + |0.1 - 0.0| = 0.1$

### 3.2 Gradients for $\hat{v}$

Similarly, the gradient for $\hat{v}$ at each pixel is:

$$
\nabla \hat{v} (x, y) = |\hat{v} (x+1, y) - \hat{v} (x, y)| + |\hat{v} (x, y+1) - \hat{v} (x, y)|
$$

For example:

- At $(0, 0): |0.0 - 0.1| + |0.0 - 0.1| = 0.2$
- At $(1, 1): |0.3 - (-0.2)| + |0.0 - (-0.2)| = 0.7$

### 3.3 Smoothness Loss

Finally, the smoothness loss is computed by averaging the gradients across all pixels. This term penalizes large discontinuities in the optical flow field, encouraging smoother motion between neighboring pixels.


## Flow Gradients for Smoothness Loss

The smoothness loss is based on the gradients of the flow components \( \hat{u} \) and \( \hat{v} \). We calculate these gradients using finite differences for each pixel in the flow field.

### Step 1: Compute Gradients for $( \hat{u} )$

For the flow component $( \hat{u} )$, the differences between neighboring pixels along both the horizontal (\( x \)) and vertical (\( y \)) directions are calculated.

#### Example flow field for $( \hat{u} )$:
$$
\hat{u} = [ 0.2, 0.0, 0.0; 0.0, 0.0, 0.1; 0.0, 0.1, 0.0 ]
$$



#### Gradient Calculation for $( \hat{u} )$:

- At $(0, 0) $:

$$
\nabla \hat{u}(0,0) = | \hat{u}_{1,0} - \hat{u}_{0,0} | + | \hat{u}_{0,1} - \hat{u}_{0,0} | = |0.0 - 0.2| + |0.0 - 0.2| = 0.2 + 0.2 = 0.4
$$

- At $(1, 0)$:

$$
\nabla \hat{u}(1,0) = | \hat{u}_{2,0} - \hat{u}_{1,0} | + | \hat{u}_{1,1} - \hat{u}_{1,0} | = |0.0 - 0.0| + |0.0 - 0.0| = 0 + 0 = 0
$$

- At $(2, 0)$:

$$
\nabla \hat{u}(2,0) = | \hat{u}_{2,0} - \hat{u}_{1,0} | + | \hat{u}_{2,1} - \hat{u}_{2,0} | = |0.0 - 0.0| + |0.0 - 0.0| = 0 + 0 = 0
$$

### Step 2: Compute Gradients for $( \hat{v} )$

Now, we compute the gradients for the \( \hat{v} \)-component.

#### Example flow field for $( \hat{v} )$:
$$
\hat{v} = [ 0.1, 0.0, 0.0; 0.0, -0.2, 0.3; 0.0, 0.0, 0.3 ]
$$



#### Gradient Calculation for $( \hat{v} )$:

- At $( (0, 0) )$:

$$
\nabla \hat{v}(0,0) = | \hat{v}_{1,0} - \hat{v}_{0,0} | + | \hat{v}_{0,1} - \hat{v}_{0,0} | = |0.0 - 0.1| + |0.0 - 0.1| = 0.1 + 0.1 = 0.2
$$

- At $( (1, 0) )$:

$$
\nabla \hat{v}(1,0) = | \hat{v}_{2,0} - \hat{v}_{1,0} | + | \hat{v}_{1,1} - \hat{v}_{1,0} | = |0.0 - 0.0| + |-0.2 - 0.0| = 0 + 0.2 = 0.2
$

- At $( (2, 0) )$:

$$
\nabla \hat{v}(2,0) = | \hat{v}_{2,0} - \hat{v}_{1,0} | + | \hat{v}_{2,1} - \hat{v}_{2,0} | = |0.0 - 0.0| + |0.0 - 0.0| = 0 + 0 = 0
$$

### Step 3: Smoothness Loss Formula

Now that we have the gradients for both the \( u \)- and \( v \)-components, the total smoothness loss is computed as:

$$
L_{\text{smooth}} = \frac{1}{N} \sum_{(x,y)} \left( \| \nabla \hat{u}(x,y) \| + \| \nabla \hat{v}(x,y) \| \right)
$$

Where $( N )$ is the number of pixels.

#### Gradients for \( \hat{u} \):
$$
\nabla \hat{u}(0,0) = 0.4, \quad \nabla \hat{u}(1,0) = 0, \quad \nabla \hat{u}(2,0) = 0 \nabla \hat{u}(0,1) = 0.4, \quad \nabla \hat{u}(1,1) = 0.1, \quad \nabla \hat{u}(2,1) = 0.1 \nabla \hat{u}(0,2) = 0.4, \quad \nabla \hat{u}(1,2) = 0, \quad \nabla \hat{u}(2,2) = 0.1
$$


#### Gradients for \( \hat{v} \):
$$
\nabla \hat{v}(0,0) = 0.2, \quad \nabla \hat{v}(1,0) = 0.2, \quad \nabla \hat{v}(2,0) = 0 \nabla \hat{v}(0,1) = 0.2, \quad \nabla \hat{v}(1,1) = 0.7, \quad \nabla \hat{v}(2,1) = 0.3 \nabla \hat{v}(0,2) = 0.2, \quad \nabla \hat{v}(1,2) = 0.3, \quad \nabla \hat{v}(2,2) = 0.3
$$



#### Summing the gradients:

$$
L_{\text{smooth}} = \frac{1}{9} \left( 0.4 + 0 + 0 + 0.4 + 0.1 + 0.1 + 0.4 + 0 + 0.1 + 0.2 + 0.2 + 0 + 0.2 + 0.7 + 0.3 + 0.2 + 0.3 + 0.3 \right)
$$

$$
L_{\text{smooth}} = \frac{1}{9} \times 4.5 = 0.5
$$

Thus, the smoothness loss for this flow field is \( L_{\text{smooth}} = 0.5 \).





### Conclusion

This example walked through the process of collecting event data, converting it into spikes, predicting optical flow, and calculating various losses including EPE, photometric consistency, and smoothness loss. These steps demonstrate how spiking neural networks can be used for event-based optical flow estimation.



### Model Architecture

1. **Starting with U-Net or Fire-FlowNet Architecture**  
   The architecture in this implementation is based on **U-Net** and **Fire-FlowNet**, which serve as the base models. We can choose one of these models based on the complexity and type of data being processed:

   #### U-Net
   - **U-Net** is a widely used architecture for image segmentation tasks, known for its encoder-decoder structure with skip connections.
   - For optical flow estimation, **U-Net** is beneficial as it captures both fine spatial details from the encoder and global context from the decoder.
   - **When to choose U-Net:** If you need to handle complex spatial data with high-level features, such as large-scale flow changes.

   #### Fire-FlowNet
   - **Fire-FlowNet** is specifically designed for optical flow estimation, especially with event-based data.
   - It utilizes a convolutional neural network (CNN) structure that operates on event frames to estimate optical flow.
   - **When to choose Fire-FlowNet:** If you are working with event-based data and require a model that is more efficient and aligned with event-driven input, offering fewer parameters and better performance for flow estimation.

2. **Incorporating Spiking Neural Networks (SNNs)**  
   To adapt **U-Net** or **Fire-FlowNet** for event-based data and spiking neurons, we follow these steps:

   #### Event Encoding
   - Convert event data into a format suitable for input into the architecture. This is done using event binning and spike encoding, where event frames (time bins) are generated to capture spiking activity over time.

   #### Spiking Neurons
   - Replace traditional neurons in the network with spiking neurons, such as the **Leaky Integrate and Fire (LIF)** model.
   - Each pixel in the event frame corresponds to a spiking neuron that accumulates spikes over time. This is crucial for optical flow tasks as it captures temporal information inherent in event-based data.

   #### Temporal Dynamics
   - Introduce learnable temporal dynamics into the spiking neurons. This allows for adaptive learning of spike timing.
   - Modify the spiking neuron layers to control how spikes evolve over time, which is essential for capturing the temporal features of event-based data.

3. **Loss Functions for Self-Supervised Learning**  
   To effectively train the model, we implement the following self-supervised loss functions:

   #### Photometric Loss
   - The **photometric loss** ensures that the predicted optical flow maintains photometric consistency between frames, reducing discrepancies between the predicted and actual frames.

   #### Smoothness Loss
   - The **smoothness loss** penalizes large variations in optical flow within a local neighborhood. This encourages smooth transitions in the flow field, essential for high-quality flow estimation.

   #### Supervised Loss (Optional)
   - If ground truth flow data is available, a **supervised loss** (such as L1 or L2 loss) can be combined with the self-supervised losses to directly compare the predicted flow with the ground truth, improving the model's accuracy.

4. **Network Training**  
   To train the adapted architecture using event-based data, follow these training strategies:

   #### Self-Supervised Training
   - If no ground truth flow is available, train the model using event frames and apply the **photometric** and **smoothness losses** to guide the learning process.

   #### Supervised Training
   - If ground truth flow data is available, use a combination of self-supervised and supervised losses to improve the flow estimation accuracy.
   - During training, ensure that the model learns the correct mapping from event data (spikes) to optical flow and optimizes the temporal dynamics of the spiking neurons.

### Conclusion
By following these steps, we can recreate the **Adaptive-SpikeNet** architecture for event-based optical flow estimation. Integrating spiking neurons with **U-Net** or **Fire-FlowNet** allows the model to effectively handle event data and learn the temporal dynamics necessary for high-quality optical flow estimation.



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
