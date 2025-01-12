import time
import torch
import torch.nn as nn
import yaml
from models.model_components.sfs_cnet_blocks import CBR, UpsampleConcat, DecoupledHead
from models.model_components.sfs_conv import SFSConv

class SFSCNet(nn.Module):
    def __init__(self, use_optimized_fpu=True):
        super(SFSCNet, self).__init__()
        loadParams()
        self.initial_cbr = CBR(3, base_channels, kernel_size=6, stride=2, padding=2)
        self.stage1_cbr = CBR(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.stage1_sfs = SFSConv(base_channels * 2, base_channels * 2, use_optimized_fpu=use_optimized_fpu)
        self.stage2_cbr = CBR(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.stage2_sfs = SFSConv(base_channels * 4, base_channels * 4, use_optimized_fpu=use_optimized_fpu)
        self.stage3_cbr = CBR(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1)
        self.stage3_sfs = SFSConv(base_channels * 8, base_channels * 8, use_optimized_fpu=use_optimized_fpu)
        self.stage4_cbr = CBR(base_channels * 8, base_channels * 16, kernel_size=3, stride=2, padding=1)
        self.stage4_sfs = SFSConv(base_channels * 16, base_channels * 16, use_optimized_fpu=use_optimized_fpu)

        # Decoder part
        self.cbr_after_stage4 = CBR(base_channels * 16, base_channels * 8, kernel_size=1, stride=1, padding=0)
        self.upsample_concat3 = UpsampleConcat(base_channels * 8, base_channels * 8)
        self.cbr_after_concat3 = CBR(base_channels * 16, base_channels * 4, kernel_size=1, stride=1, padding=0)
        self.upsample_concat2 = UpsampleConcat(base_channels * 4, base_channels * 4)
        self.cbr_after_concat2 = CBR(base_channels * 8, base_channels * 4, kernel_size=1, stride=1, padding=0)

        # Decoupled Head with upsampling to match the input image size
        self.head = DecoupledHead(num_classes=num_classes, input_channels=base_channels * 4, image_size=image_size)

    def forward(self, x):
        start_time = time.time()

        x = self.initial_cbr(x)
        #print(f"Step: Initial CBR completed in {time.time() - start_time:.4f} seconds.")

        stage1_start = time.time()
        x1 = self.stage1_cbr(x)
        x1 = self.stage1_sfs(x1)
        #print(f"Step: Stage 1 completed in {time.time() - stage1_start:.4f} seconds.")

        stage2_start = time.time()
        x2 = self.stage2_cbr(x1)
        x2 = self.stage2_sfs(x2)
        #print(f"Step: Stage 2 completed in {time.time() - stage2_start:.4f} seconds.")

        stage3_start = time.time()
        x3 = self.stage3_cbr(x2)
        x3 = self.stage3_sfs(x3)
        #print(f"Step: Stage 3 completed in {time.time() - stage3_start:.4f} seconds.")

        stage4_start = time.time()
        x4 = self.stage4_cbr(x3)
        x4 = self.stage4_sfs(x4)
        #print(f"Step: Stage 4 completed in {time.time() - stage4_start:.4f} seconds.")

        after_stage4_start = time.time()
        x4 = self.cbr_after_stage4(x4)
        #print(f"Step: CBR after Stage 4 completed in {time.time() - after_stage4_start:.4f} seconds.")

        upsample_concat3_start = time.time()
        x3_fused = self.upsample_concat3(x4, x3)
        x3_fused = self.cbr_after_concat3(x3_fused)
        #print(f"Step: Upsample and Concatenation 3 completed in {time.time() - upsample_concat3_start:.4f} seconds.")

        upsample_concat2_start = time.time()
        x2_fused = self.upsample_concat2(x3_fused, x2)
        x2_fused = self.cbr_after_concat2(x2_fused)
        #print(f"Step: Upsample and Concatenation 2 completed in {time.time() - upsample_concat2_start:.4f} seconds.")

        head_start = time.time()
        pred_bboxes, pred_classes = self.head(x2_fused)

        #print(f"Step: Decoupled Head completed in {time.time() - head_start:.4f} seconds.")
        total_elapsed_time = time.time() - start_time
        #print(f"Total forward pass completed in {total_elapsed_time:.4f} seconds.")
        return pred_bboxes, pred_classes


def loadParams(config_path="../config.yaml"):
    global num_classes, base_channels, image_size
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    num_classes = config["model"]["num_classes"]
    base_channels = config["model"]["base_channels"]
    image_size = tuple(config["training"]["input_size"])


if __name__ == "__main__":
    loadParams()
    model = SFSCNet()
    x = torch.randn(1, 3, 640, 640)
    bboxes, classes = model(x)
    print("Bounding Boxes Shape:", bboxes.shape)
    print("Classes Shape:", classes.shape)