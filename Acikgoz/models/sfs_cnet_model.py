import torch
import torch.nn as nn
import yaml

from models.model_components.sfs_cnet_blocks import CBR, UpsampleConcat, DecoupledHead
from models.model_components.sfs_conv import SFSConv

class SFSCNet(nn.Module):
    def __init__(self):
        super(SFSCNet, self).__init__()
        loadParams()
        self.initial_cbr = CBR(3, base_channels, kernel_size=6, stride=2, padding=2)
        self.stage1_cbr = CBR(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.stage1_sfs = SFSConv(base_channels * 2, base_channels * 2)
        self.stage2_cbr = CBR(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.stage2_sfs = SFSConv(base_channels * 4, base_channels * 4)
        self.stage3_cbr = CBR(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1)
        self.stage3_sfs = SFSConv(base_channels * 8, base_channels * 8)
        self.stage4_cbr = CBR(base_channels * 8, base_channels * 16, kernel_size=3, stride=2, padding=1)
        self.stage4_sfs = SFSConv(base_channels * 16, base_channels * 16)

        # Corrected part
        self.cbr_after_stage4 = CBR(base_channels * 16, base_channels * 8, kernel_size=1, stride=1, padding=0)
        self.upsample_concat3 = UpsampleConcat(base_channels * 8, base_channels * 8)
        self.cbr_after_concat3 = CBR(base_channels * 16, base_channels * 4, kernel_size=1, stride=1, padding=0)
        self.upsample_concat2 = UpsampleConcat(base_channels * 4, base_channels * 4)
        self.cbr_after_concat2 = CBR(base_channels * 8, base_channels * 4, kernel_size=1, stride=1, padding=0)
        self.head = DecoupledHead(base_channels * 4, num_classes)

    def forward(self, x):
        x = self.initial_cbr(x)
        x1 = self.stage1_cbr(x)
        x1 = self.stage1_sfs(x1)
        x2 = self.stage2_cbr(x1)
        x2 = self.stage2_sfs(x2)
        x3 = self.stage3_cbr(x2)
        x3 = self.stage3_sfs(x3)
        x4 = self.stage4_cbr(x3)
        x4 = self.stage4_sfs(x4)

        x4 = self.cbr_after_stage4(x4)
        x3_fused = self.upsample_concat3(x4, x3)
        x3_fused = self.cbr_after_concat3(x3_fused)
        x2_fused = self.upsample_concat2(x3_fused, x2)
        x2_fused = self.cbr_after_concat2(x2_fused)

        bboxes, classes = self.head(x2_fused)
        return bboxes, classes

def loadParams(config_path="../config.yaml"):
    global num_classes, base_channels
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    num_classes = config["model"]["num_classes"]
    base_channels = config["model"]["base_channels"]

if __name__ == "__main__":
    loadParams()
    model = SFSCNet()
    x = torch.randn(1, 3, 640, 640)
    bboxes, classes = model(x)
    print("Bounding Boxes Shape:", bboxes.shape)
    print("Classes Shape:", classes.shape)

