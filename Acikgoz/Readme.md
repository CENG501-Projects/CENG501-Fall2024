@TODO: Unleashing Channel Potential: Space-Frequency Selection Convolution for SAR Object Detection

This README file is an outcome of the CENG501 (Spring 2024) project for reproducing a paper without an implementation. See CENG501 (Spring 42) Project List for a complete list of all paper reproduction projects.

# 1. Introduction

This project aims to reproduce and evaluate the paper “Unleashing Channel Potential: Space-Frequency Selection Convolution for SAR Object Detection”, published in CVPR 2024. The paper introduces the SFS-Conv module, a novel convolutional architecture that addresses the challenges of SAR object detection, such as high-resolution images, small objects, and significant noise (e.g., speckle noise). SAR images, commonly used in applications like ocean monitoring, resource exploration, and disaster investigation, demand efficient and robust detection algorithms.

The primary goal of this project is to:

	1.	Implement the SFS-Conv module and the SFS-CNet architecture using PyTorch.
	2.	Validate the claims of the paper by comparing performance on datasets like HRSID, SAR-Aircraft-1.0, and SSDD.
	3.	Explore the benefits and limitations of the proposed approach to assess its practical usability and efficiency.

The implementation not only aims to faithfully replicate the original method but also addresses any ambiguities or gaps in the paper by making reasonable assumptions and interpretations.

# 1.1. Paper Summary

Problem Definition

SAR object detection faces unique challenges due to the nature of SAR imaging, which often includes small-scale objects, noisy backgrounds. Traditional convolutional architectures tend to generate redundant features, which reduces efficiency and performance. To overcome these issues, the paper introduces the Space-Frequency Selection Convolution (SFS-Conv), specifically designed for SAR object detection.

Key Contributions

	1.	SFS-Conv Module:
	•	Utilizes a shunt-perceive-select strategy to enhance feature diversity and reduce redundancy:
	•	Shunt: Splits input features into spatial and frequency components.
	•	Perceive: Extracts meaningful spatial and frequency features using SPU (Spatial Perception Unit) and FPU (Frequency Perception Unit).
	•	Select: Combines these features adaptively using a parameter-free Channel Selection Unit (CSU).
	2.	SFS-CNet: A lightweight SAR detection network based on SFS-Conv, achieving superior accuracy and efficiency compared to state-of-the-art models.
	3.	Performance Benchmarks:
	•	Demonstrates significant improvements over SoTA models on datasets like HRSID (96.2% accuracy), SAR-Aircraft-1.0 (89.7% mAP), and SSDD (99.6% AP50), while requiring fewer parameters and FLOPs.

This method integrates spatial and frequency dimensions into a single convolutional layer, unlike existing methods that rely on additional modules, reducing redundancy and computational cost.

# 2. The Method and Our Interpretation

# 2.1. The Original Method

The method proposed in the paper centers around the SFS-Conv module, which improves the representation capacity of convolutional layers by focusing on spatial and frequency features. The module is divided into three key components:

# 1. Shunt Mechanism

The input feature maps are split into two components:

	•	Spatial Features: Provide spatial information about the objects and their surroundings.
	•	Frequency Features: Capture variations in texture and reduce the impact of speckle noise using the Fractional Gabor Transform (FrGT).

# 2. Perceive Mechanism

The Perceive stage refines features using:

	•	Spatial Perception Unit (SPU):
	•	Utilizes multi-scale, dynamically adjustable convolutional kernels to capture object context, shape, and orientation.
	•	Features hierarchical residual connections to expand the receptive field effectively.
	•	Frequency Perception Unit (FPU):
	•	Uses Fractional Gabor Kernels derived from FrGT to extract multi-scale texture and orientation information.
	•	Suppresses noise and captures high-frequency features, enhancing feature diversity.

# 3. Select Mechanism

The Channel Selection Unit (CSU) adaptively fuses spatial and frequency features. It uses global average pooling and soft attention to compute channel-wise importance weights, ensuring that only the most discriminative features are preserved.

Final Integration: SFS-CNet

The SFS-CNet architecture stacks SFS-Conv modules into a lightweight object detection framework. It includes:

	•	A backbone network for feature extraction.
	•	Downsampling and upsampling layers for multi-scale detection.
	•	A Gradient-Induced Learning (OGL) strategy during training to emphasize object-level texture details.

# 2.2. Our Interpretation

Addressing Ambiguities

The paper, while comprehensive, leaves certain implementation details unclear, particularly around the following aspects:

	1.	Fractional Gabor Transform (FrGT):
	•	The FrGT’s integration with convolutional kernels is not fully detailed.
	•	We interpreted this as a separable filter operation implemented in PyTorch, ensuring rotation and scale invariance.
	2.	SPU Configuration:
	•	The exact multi-scale kernel sizes and residual connection structures were not described.
	•	We assumed progressively increasing kernel sizes (e.g., 3x3, 5x5, 7x7) with residual connections to expand the receptive field efficiently.
	3.	Feature Fusion in CSU:
	•	The paper uses “parameter-free” fusion without clear implementation steps.
	•	We interpreted this as a soft attention mechanism where spatial and frequency feature maps are weighted and summed.

Design Decisions

	•	Implementation in PyTorch:
	•	SPU and FPU were implemented as modular layers for flexibility.
	•	FrGT was implemented using 2D discrete convolution with parameterized Gabor filters.
	•	Hyperparameter Tuning:
	•	Spatial-to-frequency shunt ratio (α) was tuned to balance spatial and frequency feature contributions.
	•	Number of fractional Gabor orientations was set to 8 for optimal feature diversity without excessive computation.

# 3. Experiments and Results

3.1. Experimental Setup

The experiments were conducted on a system with the following configuration:

	•	GPU: NVIDIA RTX 4070 Ti (12 GB VRAM)
	•	CPU: AMD Ryzen 5 3600X Processor
	•	RAM: 32 GB
	•	Framework: PyTorch

The paper’s datasets, such as HRSID, SAR-Aircraft-1.0, and SSDD, were used to evaluate performance. Hyperparameters such as learning rate, weight decay, and training epochs were configured as per the paper.

Code Structure:

project/
│
├── models/
│   ├── sfs_conv.py        # Implementation of the SFS-Conv module
│   ├── sfs_cnet.py        # Full SFS-CNet model
│
├── datasets/
│   ├── hrsid_data         # HRSID Data
│   ├── hrdsid_loader.py   # Dataset loader for HRSID
│
├── experiments/
│   ├── train.py           # Training script
│   ├── test.py            # Testing and evaluation script
│
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── config.yaml            # Configuration file for hyperparameters

# 3.2. Running the Code

The implementation is provided as a PyTorch model and can be executed with the following steps:

1.	Clone the repository.
2.	Install dependencies using:
pip install -r requirements.txt
3.	Prepare the datasets and configure paths in config.yaml.
4.	Train the model:
python experiments/train.py --config config.yaml
5.	Evaluate the model:
python experiments/test.py --config config.yaml

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

Gökberk Açıkgöz, gacikgoz97@gmail.com, Middle East Technical University, Turkish Aerospace.
# Dataset
https://www.kaggle.com/datasets/sarribere99/high-resolution-sar-images-dataset-hrsid