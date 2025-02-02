import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math
from torch.nn.init import kaiming_normal_, constant_
from utils.utils import predict_flow, crop_like, conv_s, conv, deconv


####### Adapted From Spike FlowNet #######
class SpikingNN(Function):
    @staticmethod
    def forward(ctx, input):
        # Save context for backward pass
        ctx.save_for_backward(input)
        # Apply thresholding
        return input.gt(1e-5).type(torch.cuda.FloatTensor)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, = ctx.saved_tensors
        # Create gradient input
        grad_input = grad_output.clone()
        grad_input[input <= 1e-5] = 0
        return grad_input

####### Adapted From Spike FlowNet #######
""" 
We have added the "leaky" Integrate and Fire Neuron
"""
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

####### Adapted From Spike FlowNet #######
class SpikingFlowNet(nn.Module):
    expansion = 1
    def __init__(self,batchNorm=True):
        super(SpikingFlowNet,self).__init__()
        self.batchNorm = False
        self.conv1   = conv_s(self.batchNorm,   4,   64, kernel_size=3, stride=2)
        self.conv2   = conv_s(self.batchNorm,  64,  128, kernel_size=3, stride=2)
        self.conv3   = conv_s(self.batchNorm, 128,  256, kernel_size=3, stride=2)
        self.conv4   = conv_s(self.batchNorm, 256,  512, kernel_size=3, stride=2)

        self.conv_r11 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r12 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r21 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r22 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)

        self.deconv3 = deconv(self.batchNorm, 512,128)
        self.deconv2 = deconv(self.batchNorm, 384+2,64)
        self.deconv1 = deconv(self.batchNorm, 192+2,4)

        self.predict_flow4 = predict_flow(self.batchNorm, 32)
        self.predict_flow3 = predict_flow(self.batchNorm, 32)
        self.predict_flow2 = predict_flow(self.batchNorm, 32)
        self.predict_flow1 = predict_flow(self.batchNorm, 32)

        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(in_channels=512, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(in_channels=384+2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(in_channels=192+2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(in_channels=68+2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  # use 3 for dt1 and 2 for dt4
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  # use 3 for dt1 and 2 for dt4
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)


    def forward(self, input, sp_threshold):
        
        # Pad input so that we can feed it to the network without any problem
        B, C, H, W, _ = input.shape

        threshold = sp_threshold

        mem_1 = torch.zeros(input.size(0), 64, int(H/2), int(W/2)).cuda()
        mem_2 = torch.zeros(input.size(0), 128, int(H/4), int(W/4)).cuda()
        mem_3 = torch.zeros(input.size(0), 256, int(H/8), int(W/8)).cuda()
        mem_4 = torch.zeros(input.size(0), 512, int(H/16), int(W/16)).cuda()

        mem_1_total = torch.zeros(input.size(0), 64, int(H/2), int(W/2)).cuda()
        mem_2_total = torch.zeros(input.size(0), 128, int(H/4), int(W/4)).cuda()
        mem_3_total = torch.zeros(input.size(0), 256, int(H/8), int(W/8)).cuda()
        mem_4_total = torch.zeros(input.size(0), 512, int(H/16), int(W/16)).cuda()

        for i in range(input.size(4)):
            input11 = input[:, :, :, :, i].cuda()
            current_1 = self.conv1(input11)
            mem_1 = mem_1 + current_1
            mem_1_total = mem_1_total + current_1
            mem_1, out_conv1 = LIF_Neuron(mem_1, threshold)

            current_2 = self.conv2(out_conv1)
            mem_2 = mem_2 + current_2
            mem_2_total = mem_2_total + current_2
            mem_2, out_conv2 = LIF_Neuron(mem_2, threshold)

            current_3 = self.conv3(out_conv2)
            mem_3 = mem_3 + current_3
            mem_3_total = mem_3_total + current_3
            mem_3, out_conv3 = LIF_Neuron(mem_3, threshold)

            current_4 = self.conv4(out_conv3)
            mem_4 = mem_4 + current_4
            mem_4_total = mem_4_total + current_4
            mem_4, out_conv4 = LIF_Neuron(mem_4, threshold)

        out_conv4 = mem_4_total
        out_conv3 = mem_3_total
        out_conv2 = mem_2_total
        out_conv1 = mem_1_total

        out_rconv11 = self.conv_r11(out_conv4)
        out_rconv12 = self.conv_r12(out_rconv11) + out_conv4
        out_rconv21 = self.conv_r21(out_rconv12)
        out_rconv22 = self.conv_r22(out_rconv21) + out_rconv12

        flow4 = self.predict_flow4(self.upsampled_flow4_to_3(out_rconv22))
        flow4_up = crop_like(flow4, out_conv3)

        out_deconv3 = crop_like(self.deconv3(out_rconv22), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3 = self.predict_flow3(self.upsampled_flow3_to_2(concat3))
        flow3_up = crop_like(flow3, out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(self.upsampled_flow2_to_1(concat2))
        flow2_up = crop_like(flow2, out_conv1)
        out_deconv1 = crop_like(self.deconv1(concat2), out_conv1)

        concat1 = torch.cat((out_conv1,out_deconv1,flow2_up),1)
        flow1 = self.predict_flow1(self.upsampled_flow1_to_0(concat1))
        
        return flow1,flow2,flow3,flow4


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

if __name__ == "__main__":
    # Initialize the model and dummy input
    model = SpikingFlowNet().cuda()
    model.eval()

    dummy_input = torch.randn(1, 4, 272, 352, 5).cuda()
    scaled_flows = model(dummy_input, 0.75)
    
    
    for item in scaled_flows:
        print(item.shape)
