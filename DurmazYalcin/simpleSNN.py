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
class SimpleSNN(nn.Module):
    def __init__(self):
        super(SimpleSNN, self).__init__()
        self.batchNorm = True  # Define batchNorm here

        # Encoding layers (example architecture)
        self.enc1 = conv_s(self.batchNorm, 4,   64, kernel_size=3, stride=2)
        self.enc2 = conv_s(self.batchNorm, 64,  128, kernel_size=3, stride=2)
        self.enc3 = conv_s(self.batchNorm, 128,  256, kernel_size=3, stride=2)
        self.enc4 = conv_s(self.batchNorm, 256,  512, kernel_size=3, stride=2)
        
        # Residual blocks
        self.residual11 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.residual12 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.residual21 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.residual22 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)

        # Decoding layers (example architecture)
        #self.dec4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
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
        self.deconv3 = deconv(self.batchNorm, 512,128)
        self.deconv2 = deconv(self.batchNorm, 384+2,64)
        self.deconv1 = deconv(self.batchNorm, 192+2,4)

        # New layer to match the channels correctly
        self.match_layer = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)

        # Additional convolution to match channels before adding skip connections
        self.channel_match_d4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x, image_resize=256, sp_threshold=0):
        threshold = sp_threshold
        image_resize = 256
        # Encoding
        print(x.shape)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # mem_1 = torch.zeros(x.size(0), 64, int(image_resize/2), int(image_resize/2))
        # mem_2 = torch.zeros(x.size(0), 128, int(image_resize/4), int(image_resize/4))
        # mem_3 = torch.zeros(x.size(0), 256, int(image_resize/8), int(image_resize/8))
        # mem_4 = torch.zeros(x.size(0), 512, int(image_resize/16), int(image_resize/16))

        # mem_1_total = torch.zeros(x.size(0), 64, int(image_resize/2), int(image_resize/2))
        # mem_2_total = torch.zeros(x.size(0), 128, int(image_resize/4), int(image_resize/4))
        # mem_3_total = torch.zeros(x.size(0), 256, int(image_resize/8), int(image_resize/8))
        # mem_4_total = torch.zeros(x.size(0), 512, int(image_resize/16), int(image_resize/16))

        # print(x.size(3))
        # for i in range(x.size(4)):
        #     input11 = x[:, :, :, :, i]

            # current_1 = self.conv1(input11)
            # mem_1 = mem_1 + current_1
            # mem_1_total = mem_1_total + current_1
            # mem_1, out_conv1 = IF_Neuron(mem_1, threshold)

            # current_2 = self.conv2(out_conv1)
            # mem_2 = mem_2 + current_2
            # mem_2_total = mem_2_total + current_2
            # mem_2, out_conv2 = IF_Neuron(mem_2, threshold)

            # current_3 = self.conv3(out_conv2)
            # mem_3 = mem_3 + current_3
            # mem_3_total = mem_3_total + current_3
            # mem_3, out_conv3 = IF_Neuron(mem_3, threshold)

            # current_4 = self.conv4(out_conv3)
            # mem_4 = mem_4 + current_4
            # mem_4_total = mem_4_total + current_4
            # mem_4, out_conv4 = IF_Neuron(mem_4, threshold)

        # mem_4_residual = 0
        # mem_3_residual = 0
        # mem_2_residual = 0

        # self.enc4 = mem_4_total + mem_4_residual
        # self.enc3 = mem_3_total + mem_3_residual
        # self.enc2 = mem_2_total + mem_2_residual
        # self.enc1 = mem_1_total

        out_rconv11 = self.residual11(e4)
        out_rconv12 = self.residual12(out_rconv11) + e4
        out_rconv21 = self.residual21(out_rconv12)
        out_rconv22 = self.residual22(out_rconv21) + out_rconv12
        # Resize out_rconv22 to match the spatial dimensions of flow4 and e3 (80x60)
        out_rconv22 = self.upsampled_flow4_to_3(out_rconv22)

        print("Shape of out_rconv22:",out_rconv22.shape)
        

        flow4 = self.flow_prediction4(self.upsampled_flow4_to_3(out_rconv22))
        flow4_up = crop_like(flow4, e3)
        print("Shape of flow4:", flow4.shape)
        print("Shape of e3:", e3.shape)

        out_deconv3 = crop_like(self.deconv3(out_rconv22), e3)
        print("Shape of out_deconv3:", out_deconv3.shape)


        print("Shape of out_deconv3:", out_deconv3.shape)

        concat3 = torch.cat((e3,out_deconv3,flow4_up),1)
        print("Shape of concat3:", concat3.shape)
        flow3 = self.flow_prediction3(self.upsampled_flow3_to_2(concat3))
        flow3_up = crop_like(flow3, e2)
        out_deconv2 = crop_like(self.deconv2(concat3), e2)

        concat2 = torch.cat((e2,out_deconv2,flow3_up),1)
        flow2 = self.flow_prediction2(self.upsampled_flow2_to_1(concat2))
        flow2_up = crop_like(flow2, e1)
        out_deconv1 = crop_like(self.deconv1(concat2), e1)

        concat1 = torch.cat((e1,out_deconv1,flow2_up),1)
        flow1 = self.flow_prediction1(self.upsampled_flow1_to_0(concat1))



        # # Residual block
        # r = self.residual(e4)
        
        # # Decoding
        # d4 = self.dec4(r)
        # d4_cropped = crop_like(d4, e3)  # Crop d4 to match e3
        # d4 = self.channel_match_d4(d4_cropped)  # Match the channels of d4 to e3 (256 channels)
        # d4 = d4 + e3  # Add skip connection from e3
        
        # d3 = self.dec3(d4)
        # d3_cropped = crop_like(d3, e2)  # Crop d3 to match e2
        # d3 = d3_cropped + e2  # Add skip connection from e2
        
        # d2 = self.dec2(d3)
        # d2_cropped = crop_like(d2, e1)  # Crop d2 to match e1
        # d2 = d2_cropped + e1  # Add skip connection from e1
        
        # d1 = self.dec1(d2)  # Output 2 channels
        # #print(d2.shape)
        # # Full-scale flow prediction
        # flow = self.flow_prediction2(d2)
        
        return flow1,flow2,flow3,flow4
        

# Initialize the model and dummy input
model = SimpleSNN()

# Dummy input with batch size 1, 1 channel, and size 640x480
dummy_input = torch.randn(1,4, 640, 480)

# Forward pass
output = model(dummy_input)

print("Output shape:", output.shape)
#print("Full scale flow shape:", full_scale_flow.shape)
