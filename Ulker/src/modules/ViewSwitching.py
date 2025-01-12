import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchsummary import summary
import scipy
from src.modules import SuperPoint
import os
import numpy as np

class ViewSwitcher(nn.Module):
    def __init__(self, threshold):
        super(ViewSwitcher, self).__init__()
        self.threshold = threshold

        # Operations
        self.adaptive_pool = torch.nn.AdaptiveAvgPool3d(output_size=(1, 20, 20))
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.CNN = nn.Sequential(
                        # First Conv-BN-ReLU-MaxPool
                        nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3, stride=1, padding=1),  # Conv layer
                        nn.BatchNorm2d(10),  # Batch normalization
                        nn.ReLU(),  # ReLU activation
                        nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pooling
                        
                        # Second Conv-BN-ReLU-MaxPool
                        nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),  # Conv layer
                        nn.BatchNorm2d(20),  # Batch normalization
                        nn.ReLU(),  # ReLU activation
                        nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pooling

                        # Adaptive Average Pooling: Spatial dimension -> 1x1
                        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                        
                        # Linear Layer
                        nn.Flatten(),
                        nn.Linear(20, 1),
                    )

        self.softmax = torch.nn.Softmax()

        # detector head
        # import weights first
        SuperPointNetwork = SuperPoint.SuperPointNet()
        model_path = os.path.join(os.getcwd(), "src\\modules\\superpoint_v1.pth")
        SuperPointNetwork.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        self.convPa = torch.nn.Conv2d(90, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=True)

        # Freeze
        # Since shape of layers could not match, superpoint detector head sizes changed and did not freezed.
        # for param in self.convPa.parameters():
        #     param.requires_grad = False
        # for param in self.convPb.parameters():
        #     param.requires_grad = False
        # Load weights
        # self.convPa.weight = SuperPointNetwork.convPa.weight
        # self.convPb.weight = SuperPointNetwork.convPb.weight



    def forward(self, feature_map1, feature_map2):
        # Pooling
        feature_map_1a = self.adaptive_pool(feature_map1)
        feature_map_2a = self.adaptive_pool(feature_map2)

        # Correlation
        cross_correlation_map = torch.matmul(feature_map_1a.reshape(1, -1, 1), feature_map_2a.reshape(-1, 1, 1))  # transpose of last 2 channels
        cross_correlation_map = cross_correlation_map.view(1, 20, 20, 400)
        proccessed_map = self.CNN(cross_correlation_map)

    
        # Comparison
        result = self.softmax(proccessed_map)
        
        if ( result > self.threshold):
            larger_scale_map, smaller_scale_map = feature_map1, feature_map2
        else:
            larger_scale_map, smaller_scale_map = feature_map2, feature_map1
        
        # Detector Head from Superpoint
        cPa = self.relu(self.convPa(larger_scale_map))
        sparse_features = self.convPb(cPa)

        return sparse_features, larger_scale_map, smaller_scale_map
