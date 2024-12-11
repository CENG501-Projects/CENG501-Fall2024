import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

class FasterRCNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = vgg16(pretrained=True).features
        backbone.out_channels = 512  # VGG16's last conv layer output

        # Define the Anchor Generator
        anchor_generator = AnchorGenerator(
            sizes=((16,),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        # Define the RoI Align pooling
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0"],  # Single feature map
            output_size=(8, 8),   # Fixed 7x7 output
            sampling_ratio=2
        )

        # Initialize FasterRCNN model
        self.FasterRCNN_Model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=box_roi_pool
        )

    def forward(self, images, targets):
        
        return self.FasterRCNN_Model(images, targets)

