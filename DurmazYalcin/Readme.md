# @TODO: Paper title

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction
An event-based camera is a new sensor modality that operates by detecting changes in pixel intensity rather than capturing entire frames. When a change in intensity occurs at any pixel, the camera reports the pixel’s coordinates, the polarity of the change (indicating whether the intensity increased or decreased), and the precise timestamp of the event. This unique approach enables the camera to achieve high temporal resolution, capturing rapid motion. It also delivers a high dynamic range, allowing it to perform effectively in scenes with extreme lighting contrasts. Furthermore, its design significantly reduces power consumption, making it an energy-efficient alternative to traditional frame-based imaging systems.

![Events Camera](https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/DurmazYalcin/Figures/output.gif)

The image above depicts a traditional image frame on the left, while the middle panel shows accumulated events over time. In the middle panel, blue represents positive polarity (increased intensity), and red represents negative polarity (decreased intensity). The rightmost panel, on the other hand, illustrates the optical flow.

The motion of each pixel between two successive images is referred to as dense optical flow. This information is crucial for numerous downstream tasks, including Simultaneous Localization and Mapping (SLAM) and odometry, where understanding motion dynamics is fundamental. While optical flow estimation from conventional image frames has been extensively studied and significantly advanced over the years, estimating optical flow from event-based cameras remains a challenging and less explored area due to the fundamentally different data representation and asynchronous nature of event streams.

A few attempts have been made to estimate optical flow from event streams using conventional neural networks, such as E-RAFT and EV-Flownet. These methods rely on accumulated events to compute the optical flow, effectively adapting traditional image-based approaches to event data. In contrast, Spike-FlowNet employs Integrate-and-Fire (IF) neurons, or spiking neurons, which are more aligned with the asynchronous and sparse nature of event streams.


## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and our interpretation

## 2.1. The original method

@TODO: Explain the original method.

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
