# Adaptive-SpikeNet: Event-based Optical Flow Estimation using Spiking Neural Networks with Learnable Neuronal Dynamics

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction
An event-based camera is a new sensor modality that operates by detecting changes in log pixel intensity rather than capturing entire frames. When a change in log intensity occurs at any pixel, the camera reports the pixel’s coordinates, the polarity of the change (indicating whether the intensity increased or decreased), and the precise timestamp of the event as a tuple {x,y,t,p}. This unique approach enables the camera to achieve high temporal resolution, capturing rapid motion. It also delivers a high dynamic range, allowing it to perform effectively in scenes with extreme lighting contrasts. Furthermore, its design significantly reduces power consumption, making it an energy-efficient alternative to traditional frame-based imaging systems.

<div align="center">
  <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/output.gif" alt="description" width="80%">
</div>

The gif above depicts a small interval from [MVSEC](https://daniilidis-group.github.io/mvsec/) dataset. 
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
<div align="center">
  <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/InputRepresentation.gif" alt="description" width="80%">
</div>

#### Scaled Flows
When backpropagation is applied at four different scales of optical flow, we adjust the motion values (optical flow) for each scale. For example, if an image is resized to be twice as large, the motion of each pixel also doubles. To ensure the motion stays accurate, we scale the optical flow values to match the size of the image at each scale. This way, the optical flow at each scale correctly represents the motion for that specific scale.

```python
def get_scaled_tensors(tensor):
    flow_tensor_arr = []
    H = tensor.shape[2]
    W  = tensor.shape[3]
    for idx in range(4):
        new_W, new_H = W // (2**idx), H // (2**idx)
        resized_tensor = F.interpolate(tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)
        flow_tensor_arr.append(resized_tensor)
    return flow_tensor_arr
```

### Integrate and Fire Neurons (Leaky)
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

To incorporate the leakage factor into our code, we introduced two terms: a decay factor $ λ = 0.9 $ and a $threshold = 0.75 $. These terms were defined as hyperparameters within the membrane potential history. The original paper also explored these parameters but concluded that they have minimal impact on the algorithm's overall performance. The implementation details can be seen as follows;

```python
def LIF_Neuron(membrane_potential, threshold):
    global threshold_k
    threshold_k = threshold
    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold_k, 0)
    membrane_potential = membrane_potential - ex_membrane # hard reset
    membrane_potential = membrane_potential * 0.9
    # generate spike
    out = SpikingNN.apply(ex_membrane)
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
### Architecture Implementation


The architecture in Figure 2 is designed to leverage the unique capabilities of event cameras for optical flow estimation. Event cameras provide asynchronous, high-temporal-resolution data, recording brightness changes as ON and OFF polarity events. These events are discretized into two spatiotemporal streams, the Former Events and the Latter Events, which represent brightness changes over two consecutive intervals. This discretization provides the foundational input for the network, encoding fine-grained motion information from the scene. On top of this discretization, we need to take into account ON and OFF events during the binning procedure. Each bin is composed of 4 frames coming from an ON event frame from Former Events, an OFF event frame from Former Events, an ON event frame from Latter Events and an OFF event frame from Latter Events.

The encoder (represented by blue blocks) is responsible for extracting hierarchical features from the input event streams. It begins with the raw data encoded into b feature channels, which are progressively compressed through a series of convolutional and downsampling operations. With each stage, the spatial resolution decreases while the number of feature channels increases (b → 2b → 4b → 8b), enabling the model to capture increasingly abstract and higher-level spatiotemporal patterns. These features encapsulate both the spatial structure and the motion dynamics inherent in the input event data, providing the foundation for accurate optical flow estimation.

Within the architecture, the residual block (orange blocks) plays a critical role in refining the feature maps generated by the encoder. This block processes the encoded features using multiple layers, each operating on 16b channels, with residual connections preserving the original input information. These connections help the network focus on learning differences between input and output rather than relearning redundant features. This refinement process ensures robust feature representation while addressing common challenges in deep networks, such as vanishing gradients and degraded performance in very deep architectures. By integrating these residuals, the architecture achieves a balance between learning complex features and maintaining computational efficiency.

The decoder (green dashed section) reconstructs the optical flow predictions by progressively increasing the spatial resolution of the feature maps. Using transposed convolutions (yellow blocks), the decoder performs upsampling in stages (8b → 4b → 2b → b), reversing the compression applied by the encoder. To enhance the reconstructed features, the decoder employs skip connections (black circles), which concatenate corresponding feature maps from the encoder stages. This mechanism reintroduces lower-level details lost during encoding, enriching the upsampled features with spatial context and fine details.

An additional component of the architecture is the spike accumulator (green arrows), which integrates multi-scale features during decoding. This process likely aggregates temporal and spatial consistency across different resolutions, improving the overall accuracy and robustness of the optical flow predictions. By leveraging multi-resolution signals, the architecture ensures that both coarse and fine motion details are well-represented, capturing the complexities of dynamic scenes.

The final output consists of full-scale flow predictions generated at multiple resolutions. This multi-resolution approach allows the network to capture fine-grained motion details while simultaneously providing a broader perspective on scene dynamics. Such a design is particularly effective in handling complex, high-speed motion scenarios, making it well-suited for tasks where precision and adaptability are essential.



## 3.2. Running the code

### Preprocessing The Data
Working with raw data in [DSEC](https://dsec.ifi.uzh.ch/) and [MVSEC](https://daniilidis-group.github.io/mvsec/) can be highly time-consuming due to the storage of events in long arrays, making the extraction of relevant events for a single iteration inefficient. To address this, we binarize the events offline before training. Preprocessing codes for both [MVSEC](https://daniilidis-group.github.io/mvsec/) and [DSEC](https://dsec.ifi.uzh.ch/) are available.

For [MVSEC](https://daniilidis-group.github.io/mvsec/), simply download the relevant data. Use the provided [preprocessing script for MVSEC](https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/MVSECUtils/preprocessMVSEC.py) for efficient binarization and preparation.

For [DSEC](https://dsec.ifi.uzh.ch/), download the dataset and ensure the data format adheres to the required structure before running [the preprocessing script for DSEC](https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/DSECUtils/preprocessDSEC.py). 
```bash
├── path_to_dataset
      ├── thun_00_a
      ├── zurich_city_01_a
      ├── zurich_city_02_a
          ├──zurich_city_02_a_optical_flow_forward_timestamps.txt
          ├── zurich_city_02_a_events_left
              ├── events.h5
              ├── rectify_map.h5
          ├── zurich_city_02_a_optical_flow_forward_event
              ├── <flow_name_0>.png
              ├── <flow_name_1>.png
              ├── <flow_name_1>.png
              ...
      ├── zurich_city_02_c
      ├── zurich_city_02_d
      ...
```

Once the preprocessed data is available, store them as training set and validation set in different paths.
```bash
├── path_to_dataset
      ├── training
          ├── Sequence 1
              ├── 00000.hdf5
              ├── 00001.hdf5
              ...
          ├── Sequence 2
          ├── Sequence 3
          ...
          ├── Sequence n
      ├── validation
          ├── Sequence n+1
          ├── Sequence n+2
          ...
```

### Training
The ground truth provided by MVSEC is less accurate compared to DSEC. To address this, **we use self-supervised loss for MVSEC**, leveraging the data without relying heavily on its less precise ground truth. Conversely, **for DSEC, we apply supervised loss**, taking advantage of its more accurate ground truth annotations. 

Additionally, we have designed two distinct networks, each with specific input format requirements. Due to these differences, separate training codes are provided for each network and for each dataset. 

* [Train EventFlow network with DSEC dataset using supervised loss.](https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/TrainEventFlowWithDSEC.py)
* [Train EventFlow network with MVSEC dataset using self-supervised loss.](https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/TrainEventFlowWithMVSEC.py)
* [Train Spiking network with DSEC dataset using supervised loss.](https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/TrainSpikeFlowWithDSEC.py)
* [Train Spiking network with MVSEC dataset using self-supervised loss.](https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/TrainSpikeFlowWithMVSEC.py)

All parameters and paths are managed in one class. Simply set the paths for training and validation data, and the scripts will handle data processing automatically. You can also adjust the batch size based on your system's capabilities.
```python
class parameters:
    def __init__(self):
        self.train_data_path    = "/media/romer/Additional/OpticalFlow/Training"
        self.valid_data_path    = "/media/romer/Additional/OpticalFlow/Validation"

        self.saving_path  = "checkpoints/MVSEC/SpikingNet" 
        self.epochs       = 50
        
        self.batch_size   = 64
        
        self.lr           = 5e-5 
        self.momentum     = 0.9
        self.weight_decay = 4e-4
        self.beta         = 0.999
```

### Inference
You can find our trained weigths [here](https://drive.google.com/drive/folders/14KGo-5k25KVVTg1SH69Qhbxw1FsrhNle?usp=sharing).

## 3.3. Results
Inside the [DSEC](https://dsec.ifi.uzh.ch/) dataset, we identified 8,211 frames with optical flow as ground truth. Of these, 8,170 frames were used for training, while the remaining 41 frames were set aside for validation. The network was trained for 30 epochs, and the training and validation losses are reported below.
<div align="center">
  <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/training_losses.png" alt="description" width="50%">
</div>

The validation loss is smaller than the training loss because the validation set includes a simpler scenario with relatively stable and moderate optical flow. In contrast, the training set contains samples with higher optical flow vectors.

Finally, we visualize sample estimations from the network, showing both the ground truth optical flow and the estimated flow. It is important to note that the ground truth is available only for a sparse set of pixels. To facilitate comparison, we apply the same mask to the estimated flow, resulting in a masked estimated flow. This masking is done purely for your convenience to directly compare the ground truth with the estimation.
<div align="center">
  <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/ESvsGT.gif" alt="description" width="95%">
</div>

According to our observations with the [DSEC](https://dsec.ifi.uzh.ch/) dataset, we trained the network with the [MVSEC](https://daniilidis-group.github.io/mvsec/) dataset for 20 epochs, and the training and validation losses are reported below.
```
Will be available later
```

# 4. Conclusion

In this study, we utilized the MVSEC and DSEC datasets, both of which provide LIDAR, IMU, event camera, and grayscale image data. Among these, the DSEC dataset offers a more reliable ground truth compared to MVSEC. For our implementation, we focused exclusively on the event camera data to estimate optical flow using the adaptive spike flow architecture outlined in the referenced paper.

Our architecture consists of three primary components: an encoder, residual connections, and a decoder. The encoder includes four cascaded and parallel working blocks to effectively extract features from the input data. This is followed by two convolutional layers with residual connections, enabling the network to retain and refine critical information. Finally, the decoder, comprising four cascaded and parallel blocks, estimates the optical flow as the output.

When working with the DSEC dataset, the availability of ground truth data allows us to perform supervised training by backpropagating the error at each stage of the decoder. In contrast, due to the lack of reliable ground truth in the MVSEC dataset, training is carried out in an unsupervised manner. Our results align with those reported in the paper, with noticeable improvements observed on the DSEC dataset. This enhancement can be attributed to the architecture being specifically designed for custom hardware optimized for neuromorphic sensors like event cameras.

Overall, this project has been a valuable learning experience, serving as an introductory step in implementing deep learning architectures. It has also broadened our perspective on leveraging event-based sensing and highlighted the potential of neuromorphic computing in optical flow estimation.

# 5.  Contact

Atakan Durmaz - atakan.durmaz@metu.edu.tr
Haktan Yalçın - yalcin.haktan@metu.edu.tr

# 6. References

```latex
@inproceedings{kosta2023adaptive,
  title={Adaptive-SpikeNet: Event-Based Optical Flow Estimation Using Spiking Neural Networks with Learnable Neuronal Dynamics},
  author={Kosta, Adarsh Kumar and Roy, Kaushik},
  booktitle={International Conference on Robotics and Automation (ICRA)},
  year={2023},
  organization={IEEE}
}

@inproceedings{lee2020spike,
  title={Spike-Flownet: Event-Based Optical Flow Estimation with Energy-Efficient Hybrid Neural Networks},
  author={Lee, Chankyu and Kosta, Adarsh Kumar and Zhu, Alex Zihao and Chaney, Kenneth and Daniilidis, Kostas and Roy, Kaushik},
  booktitle={European Conference on Computer Vision},
  pages={366--382},
  year={2020},
  organization={Springer}
}
```



