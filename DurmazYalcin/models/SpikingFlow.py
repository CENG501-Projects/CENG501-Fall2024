import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math
import snntorch as snn
import sys

class paramters:
    def __init__(self):
        self.beta_init = 1.0

class SpikingNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        self.params = params

        # ENCODER LAYERS
        self.encoder1 = self.encoder(4,64)
        self.encoder1_lif = snn.Leaky(beta=self.params.beta_init, learn_beta=True, learn_threshold=True)
        
        self.encoder2 = self.encoder(64,128)
        self.encoder2_lif = snn.Leaky(beta=self.params.beta_init, learn_beta=True, learn_threshold=True)
        
        self.encoder3 = self.encoder(128,256)
        self.encoder3_lif = snn.Leaky(beta=self.params.beta_init, learn_beta=True, learn_threshold=True)
        
        self.encoder4 = self.encoder(256,512)
        self.encoder4_lif = snn.Leaky(beta=self.params.beta_init, learn_beta=True, learn_threshold=True)


        # RESIDUAL LAYERS
        self.residual11 = self.resnet(512,512)
        self.residual11_lif = snn.Leaky(beta=self.params.beta_init, learn_beta=True, learn_threshold=True)
        
        self.residual12 = self.resnet(512,512)
        self.residual12_lif = snn.Leaky(beta=self.params.beta_init, learn_beta=True, learn_threshold=True)
        
        self.residual21 = self.resnet(512,512)
        self.residual21_lif = snn.Leaky(beta=self.params.beta_init, learn_beta=True, learn_threshold=True)
        
        self.residual22 = self.resnet(512,512)
        self.residual22_lif = snn.Leaky(beta=self.params.beta_init, learn_beta=True, learn_threshold=True)
        
        
        # DECODER LAYERS
        self.decoder4  = self.decoder(512, 256)
        self.decoder4_lif = snn.Leaky(beta=self.params.beta_init, learn_beta=True, learn_threshold=True)
        
        self.decoder3  = self.decoder(256+2, 128)
        self.decoder3_lif = snn.Leaky(beta=self.params.beta_init, learn_beta=True, learn_threshold=True)
        
        self.decoder2  = self.decoder(128+2, 64)
        self.decoder2_lif = snn.Leaky(beta=self.params.beta_init, learn_beta=True, learn_threshold=True)
        
        self.decoder1  = self.decoder(64+2, 32)
        self.decoder1_lif = snn.Leaky(beta=self.params.beta_init, learn_beta=True, learn_threshold=True)

        # FLOW PREDICTORS
        self.flow3 =  self.predict_flow(256)
        self.flow2 =  self.predict_flow(128)
        self.flow1 =  self.predict_flow(64)
        self.flow  =  self.predict_flow(32)
        
        # FINAL FLOWS
        self.final_flow  = self.resnet(10,2)
        
    def forward(self, x):
        ###### INITIALIZE THE MEMBRANE POTANTIALS ######
        # Encoder
        enc1_mem = self.encoder1_lif.init_leaky()
        enc2_mem = self.encoder2_lif.init_leaky()
        enc3_mem = self.encoder3_lif.init_leaky()
        enc4_mem = self.encoder4_lif.init_leaky()
        
        # Residual Net
        res11_mem = self.residual11_lif.init_leaky()
        res12_mem = self.residual12_lif.init_leaky()
        res21_mem = self.residual21_lif.init_leaky()
        res22_mem = self.residual22_lif.init_leaky()
        
        # Decoder
        dec3_mem = self.decoder4_lif.init_leaky()
        dec2_mem = self.decoder3_lif.init_leaky()
        dec1_mem = self.decoder2_lif.init_leaky()
        dec0_mem = self.decoder1_lif.init_leaky()
        
        # Middle step flows
        flows3 = []
        flows2 = []
        flows1 = []
        flows  = []

        for step in range(x.size(0)):  # Time steps for spiking inputs
            # Get the input
            x_t = x[step]

            # Encoder
            enc1 = self.encoder1(x_t)
            enc1_spk, enc1_mem = self.encoder1_lif(enc1, enc1_mem)

            enc2 = self.encoder2(enc1_spk)
            enc2_spk, enc2_mem = self.encoder2_lif(enc2, enc2_mem)

            enc3 = self.encoder3(enc2_spk)
            enc3_spk, enc3_mem = self.encoder3_lif(enc3, enc3_mem)

            enc4 = self.encoder4(enc3_spk)
            enc4_spk, enc4_mem = self.encoder4_lif(enc4, enc4_mem)

            # Residual Network
            res11 = self.residual11(enc4_spk)
            res11_spk, res11_mem = self.residual11_lif(res11, res11_mem)
 
            res12 = self.residual12(res11_spk + enc4_spk)   # We have a skip connection!!!
            res12_spk, res12_mem = self.residual12_lif(res12, res12_mem)

            res21 = self.residual21(res12_spk)
            res21_spk, res21_mem = self.residual21_lif(res21, res21_mem)

            res22 = self.residual22(res21_spk + res12)      # We have a skip connection!!!
            res22_spk, res22_mem = self.residual22_lif(res22, res22_mem)

            # Decoder
            dec3  = self.decoder4(res22_spk)
            dec3_spk, dec3_mem = self.decoder4_lif(dec3, dec3_mem)
            flow3  = self.flow3(dec3_spk)  # Estimate the optical flow
            flows3.append(flow3)
            
            concatted3 = torch.cat((dec3_spk,flow3),1)
            
            dec2  = self.decoder3(concatted3)
            dec2_spk, dec2_mem = self.decoder3_lif(dec2, dec2_mem)
            flow2  = self.flow2(dec2_spk)  # Estimate the optical flow
            flows2.append(flow2)
            
            concatted2 = torch.cat((dec2_spk,flow2),1)

            dec1  = self.decoder2(concatted2)
            dec1_spk, dec1_mem = self.decoder2_lif(dec1, dec1_mem)
            flow1  = self.flow1(dec1_spk)  # Estimate the optical flow
            flows1.append(flow1)
            
            concatted1 = torch.cat((dec1_spk,flow1),1)

            dec0  = self.decoder1(concatted1)
            dec0_spk, dec0_mem = self.decoder1_lif(dec0, dec0_mem)

            flow   = self.flow(dec0_spk)  # Estimate the optical flow
            flows.append(flow)
        
        concatenated_flows3 = torch.cat(flows3, dim=1)
        concatenated_flows2 = torch.cat(flows2, dim=1)
        concatenated_flows1 = torch.cat(flows1, dim=1)
        concatenated_flows  = torch.cat(flows, dim=1)
        
        final_flow3 = self.final_flow(concatenated_flows3)
        final_flow2 = self.final_flow(concatenated_flows2)
        final_flow1 = self.final_flow(concatenated_flows1)
        final_flow  = self.final_flow(concatenated_flows)

        return final_flow, final_flow1, final_flow2, final_flow3
        
    def encoder(self, layers_in, layers_out):
        net = nn.Sequential(
            nn.Conv2d(layers_in, layers_out, kernel_size=3, stride=2, padding=2//2, bias=False),
        )
        return net
    
    def resnet(self, layers_in, layers_out):
        net = nn.Sequential(
            nn.Conv2d(layers_in, layers_out, kernel_size=3, stride=1, padding=2//2, bias=False),
        )
        return net
    
    def decoder(self, layers_in, layers_out):
        net = nn.Sequential(
            nn.ConvTranspose2d(layers_in, layers_out, kernel_size=4, stride=2, padding=2//2, bias=False),
        )
        return net
        
    def predict_flow(self, layers_in):
        layer = nn.Sequential(
                    nn.Conv2d(layers_in, 2, kernel_size=1, stride=1, padding=0, bias=False)
                )
        return layer
    
    
if __name__ == "__main__":
    # Instantiate the model
    params = paramters()
    model = SpikingNet(params)
    
    # Create a dummy input tensor (times=10, batch_size=1, channels=4, height=28, width=28)
    dummy_input = torch.randn(5, 3, 4, 480, 640)

    # Pass the dummy input through the network
    estimated_flows = model(dummy_input)
    
    for item in estimated_flows:
        print(item.shape)
    
