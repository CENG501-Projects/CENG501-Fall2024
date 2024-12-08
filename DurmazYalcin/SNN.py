import torch
import torch.nn as nn
import torch.nn.functional as F

# Adjusted crop_like function as per your suggestion
def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]

def conv_s(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
        )
    
    
def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        
def deconv(batchNorm, in_planes, out_planes):
    if batchNorm:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True))


def predict_flow(batchNorm, in_planes):
    if batchNorm:
        return nn.Sequential(
                nn.BatchNorm2d(32),
                nn.Conv2d(in_planes,2,kernel_size=1,stride=1,padding=0,bias=False),
            )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, 2, kernel_size=1, stride=1, padding=0, bias=False),
        )
        
    
# Define the simple SNN model
class SpikeFlow(nn.Module):
    def __init__(self):
        super(SpikeFlow, self).__init__()
        self.batchNorm = False  # Define batchNorm here

        # Encoding layers (example architecture)
        self.enc1 = conv_s(self.batchNorm, 20,   64, kernel_size=3, stride=2)
        self.enc2 = conv_s(self.batchNorm, 64,  128, kernel_size=3, stride=2)
        self.enc3 = conv_s(self.batchNorm, 128,  256, kernel_size=3, stride=2)
        self.enc4 = conv_s(self.batchNorm, 256,  512, kernel_size=3, stride=2)
        
        # Residual blocks
        self.residual11 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.residual12 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.residual21 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.residual22 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)

        # Decoding layers (example architecture)
        self.dec3 = deconv(self.batchNorm, 512, 128)
        self.dec2 = deconv(self.batchNorm, 384+2,64)
        self.dec1 = deconv(self.batchNorm, 6192+2,4)
        
        
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(in_channels=512, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(in_channels=384+2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(in_channels=192+2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(in_channels=68+2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)

        # Flow predictions
        self.flow_prediction1 = predict_flow(self.batchNorm, 32)
        self.flow_prediction2 = predict_flow(self.batchNorm, 32)
        self.flow_prediction3 = predict_flow(self.batchNorm, 32)
        self.flow_prediction4 = predict_flow(self.batchNorm, 32)

        #Deconvolution layers
        self.deconv3 = deconv(self.batchNorm, 512, 128)
        self.deconv2 = deconv(self.batchNorm, 384+2,64)
        self.deconv1 = deconv(self.batchNorm, 192+2,4)

        # New layer to match the channels correctly
        self.match_layer = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)

        # Additional convolution to match channels before adding skip connections
        self.channel_match_d4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        """ Data Flow and Shapes
        # Input
        x:      torch.Size([1, 4, 640, 480])
        
        # Encoder
        e1:     torch.Size([1, 64, 320, 240])
        e2:     torch.Size([1, 128, 160, 120])
        e3:     torch.Size([1, 256, 80, 60])
        e4:     torch.Size([1, 512, 40, 30])
        
        # Residual Network
        rxx:    torch.Size([1, 512, 40, 30])
        
        # Decoder
        d4:         torch.Size([1, 32, 80, 60])
        flow4:      torch.Size([1, 2, 80, 60])
        flow4_up:   torch.Size([1, 2, 80, 60])
        
        deconv3:    torch.Size([1, 128, 80, 60])
        concat3:    torch.Size([1, 386, 80, 60])
        d3:         torch.Size([1, 32, 160, 120])
        flow3:      torch.Size([1, 2, 160, 120])
        flow3_up:   torch.Size([1, 2, 160, 120])
        
        deconv2:    torch.Size([1, 64, 160, 120])
        concat2:    torch.Size([1, 194, 160, 120])
        d2:         torch.Size([1, 32, 320, 240])
        flow2:      torch.Size([1, 2, 320, 240])
        flow2_up:   torch.Size([1, 2, 320, 240])
        
        deconv1:    torch.Size([1, 4, 320, 240])
        concat1:    torch.Size([1, 70, 320, 240])
        d1:         torch.Size([1, 32, 640, 480])
        flow1:      torch.Size([1, 2, 640, 480])
        """
        
        ########### ENCODER ###########
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
      
        ########### RESIDUAL ###########
        r11 = self.residual11(e4)
        r12 = self.residual12(r11) + e4
        r21 = self.residual21(r12)
        r22 = self.residual22(r21) + r12
        
        ########### DECODER ###########
        d4 = self.upsampled_flow4_to_3(r22)
        flow4 = self.flow_prediction4(d4)
        flow4_up = crop_like(flow4, e3)

        deconv3 = self.deconv3(r22)
        deconv3 = crop_like(deconv3, e3)
        concat3 = torch.cat((e3,deconv3,flow4_up),1)
        d3 = self.upsampled_flow3_to_2(concat3)
   
        flow3 = self.flow_prediction3(d3)
        flow3_up = crop_like(flow3, e2)
        
        deconv2 = self.deconv2(concat3)
        deconv2 = crop_like(deconv2, e2)
        concat2 = torch.cat((e2,deconv2,flow3_up),1)
        
        d2 = self.upsampled_flow2_to_1(concat2)
        flow2 = self.flow_prediction2(d2)
        flow2_up = crop_like(flow2, e1)
        

        
        deconv1 = self.deconv1(concat2)
        deconv1 = crop_like(deconv1, e1)
        concat1 = torch.cat((e1,deconv1,flow2_up),1)
        
        d1 = self.upsampled_flow1_to_0(concat1)
        flow1 = self.flow_prediction1(d1)

        return flow1,flow2,flow3,flow4
        

if __name__ == "__main__":
    # Initialize the model and dummy input
    model = SpikeFlow().cuda()
    model.eval()

    # Dummy input with batch size 1, 1 channel, and size 640x480
    dummy_input = torch.randn(1,20, 480, 640).cuda()

    # Forward pass
    output = model(dummy_input)
