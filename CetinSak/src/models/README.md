# Model Implementations

This folder holds implementations of networks used in this paper. Implementation are from [diffeo-sota repository](https://github.com/leonardopetrini/diffeo-sota/tree/main/models). Implementations of VGG, ResNet and EfficientNetB0-2 are imported to corresponding files.

Following table is implemented from Table 1 of the paper.

| **Structures**         | **VGG**                     | **ResNet**               | **EfficientNetB0-2**         |
|-------------------------|-----------------------------|--------------------------|------------------------------|
| Citation               | [Simonyan, 2015](https://arxiv.org/abs/1409.1556)    | [He, 2016](https://arxiv.org/abs/1512.03385)        | [Tan, 2019](https://arxiv.org/abs/1905.11946)           |
| **Depth**              | 11, 16                     | 18, 34                   | 18                           |
| **Num. Parameters**    | 9-15 M                     | 11-21 M                  | 5 M                          |
| **FC Layers**          | 1                          | 1                        | 1                            |
| **Activation**         | ReLU                       | ReLU                     | Swish                        |
| **Pooling**            | Max                        | Avg. (last layer only)   | Avg. (last layer only)       |
| **Dropout**            | /                          | /                        | Yes + DropConnect            |
| **Batch Norm**         | If "bn" in name            | Yes                      | Yes                          |
| **Skip Connections**   | /                          | Yes                      | Yes (inv. residuals)         |
