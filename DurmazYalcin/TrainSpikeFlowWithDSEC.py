import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from models.SpikingFlow import SpikingNet
from DSECUtils.DSECLoader import DSECManipulator
import cv2
import numpy as np
from utils.utils import padding

import os
import time
import sys


class parameters:
    def __init__(self):
        # Path to training and validation data
        self.train_data_path = "/media/hakito/HDD/event_data/dsec/ProcessedDSEC/training"
        self.valid_data_path = "/media/hakito/HDD/event_data/dsec/ProcessedDSEC/validation"
        
        # Where the save the output
        self.saving_path  = "checkpoints/SpikingNet/DSEC" 
        
        # Number of training epoches
        self.epochs       = 50
        
        # Batch size
        self.batch_size   = 1
        
        # Optimization parameters
        self.lr           = 1e-5
        self.momentum     = 0.9
        self.weight_decay = 4e-4
        self.beta         = 0.999
        
        # Spiking Network initial leakage factor
        self.beta_init = 1.0    # 1 means no leakage
        

def get_scaled_validty_tensors(validity: np.array):
    validity_tensor_arr = []
    old_height = validity.shape[1]
    old_width  = validity.shape[2]

    for idx in range(4):
        new_height = int(old_height / (2**idx))
        new_width  = int(old_width / (2**idx))
        validty_tensor = torch.from_numpy(cv2.resize(validity, (new_width,new_height), interpolation=cv2.INTER_LINEAR)).float().cuda().unsqueeze(0).unsqueeze(0)

        validty_tensor = torch.cat([validty_tensor, validty_tensor], dim=1)
        validity_tensor_arr.append(validty_tensor)
    return validity_tensor_arr


def get_scaled_tensors(tensor):
    flow_tensor_arr = []
    H = tensor.shape[2]
    W  = tensor.shape[3]
    for idx in range(4):
        new_W, new_H = W // (2**idx), H // (2**idx)
        resized_tensor = F.interpolate(tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)
        flow_tensor_arr.append(resized_tensor)
    return flow_tensor_arr


def computeLoss(flow_gt, estimated_scaled_flows):
    # Create scaled masks and groundtruth
    scaled_gt_flows    = get_scaled_tensors(flow_gt)
    loss = 0.0
    for idx in range(len(estimated_scaled_flows)):
        estimated_flow  = estimated_scaled_flows[idx]
        flow_gt         = scaled_gt_flows[idx][:,:2,:,:]
        mask            = scaled_gt_flows[idx][:,2,:,:]
 
        mask = torch.cat([mask.unsqueeze(1), mask.unsqueeze(1)], dim=1)

        discrepeancy = (flow_gt - estimated_flow)
        masked_discrepeancy = torch.mul(discrepeancy,mask)
        loss = loss + torch.mul(masked_discrepeancy,masked_discrepeancy).sum() / mask.sum()
    return loss

        
class TrainingManager:
    def __init__(self):
        # Load parameters
        self.params     = parameters()
        
        # Initialize the model
        self.model      = SpikingNet(self.params).cuda()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.params.lr, betas=(self.params.momentum, self.params.beta))
        
        # Load the training and validation dataset
        self.train_data = DataLoader( DSECManipulator(self.params.train_data_path)   , batch_size=self.params.batch_size )
        self.valid_data = DataLoader( DSECManipulator(self.params.valid_data_path), batch_size=self.params.batch_size )
        
        # Make sure that the saving path is available
        if not os.path.exists(self.params.saving_path):
            os.makedirs(self.params.saving_path)
            
        # Save the losses as well
        self.losses = open(os.path.join(self.params.saving_path, "losses.txt"),"w")
        self.losses.write("# epoch, training loss, validation loss, epoch duration (sec)\n")
        self.losses.close()
        
        
        print(f"Len of training data : {len(self.train_data)}")
        print(f"Len of validation data : {len(self.valid_data)}")

    def train(self):
        best_val_loss = 1e10
        for epoch in range(self.params.epochs):
            t_start = time.monotonic_ns()
            ################# Training #################
            train_loss = 0.0

            for idx, item in enumerate(self.train_data):
    
                # Get events
                binarized_events = item['left_events']
                
                input_array = []
                for sample in binarized_events:
                    padded_sample = padding(sample)
                    input_array.append(padded_sample)

                # Get the size of the original image
                _, _, H, W = sample.shape
                    
                # Prepare the input
                input = torch.stack(input_array, dim=0).float().cuda()

                # Get the estimated scaled flows
                estimated_scaled_flows_padded = self.model(input)
                
                # Get rid of the padded parts
                estimated_scaled_flows = []
                for scale_idx, padded_estimated_scale in enumerate(estimated_scaled_flows_padded):
                    new_H = H//(2**scale_idx)
                    new_W = W//(2**scale_idx)
                    scaled_flow = padded_estimated_scale[:,:, :new_H, :new_W]
                    estimated_scaled_flows.append(scaled_flow)
                    
                # Get the groundturth and validty mask
                flow_gt = item['optical_flow'].cuda()
                
                # Compute the loss
                loss = computeLoss(flow_gt, estimated_scaled_flows)
                train_loss += loss.item()

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Clean up memory
                del input, estimated_scaled_flows, flow_gt, loss  # Free tensors
                torch.cuda.empty_cache()
                break
                
                
            train_loss /= len(self.train_data)
            ################# Validation #################
            with torch.no_grad():
                val_loss = 0.0
                for item in self.valid_data:
                    # Get events
                    binarized_events = item['left_events']
                    
                    input_array = []
                    for sample in binarized_events:
                        padded_sample = padding(sample)
                        input_array.append(padded_sample)

                    # Prepare the input
                    input = torch.stack(input_array, dim=0).float().cuda()
                    
                    # Get the estimated scaled flows
                    estimated_scaled_flows_padded = self.model(input)
                    
                    # Get rid of the padded parts
                    estimated_scaled_flows = []
                    for scale_idx, padded_estimated_scale in enumerate(estimated_scaled_flows_padded):
                        new_H = H//(2**scale_idx)
                        new_W = W//(2**scale_idx)
                        scaled_flow = padded_estimated_scale[:,:, :new_H, :new_W]
                        estimated_scaled_flows.append(scaled_flow)
                        
                    # Get the groundturth and validty mask
                    flow_gt = item['optical_flow'].cuda()
                    
                    # Compute the loss
                    loss = computeLoss(flow_gt, estimated_scaled_flows)
                    val_loss += loss.item()
                    
                    # Clean up memory
                    del input, estimated_scaled_flows, flow_gt, loss  # Free tensors
                    torch.cuda.empty_cache()
                    print("heree")
                val_loss /= len(self.valid_data)
            
            # Print the epoch details
            t_stop = time.monotonic_ns()
            elapsed_time_sec = (t_stop - t_start) * (1e-9)
            print(f"Epoch:{epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}. Elapsed Time : {elapsed_time_sec} sec")
            
            # Save the results into txt
            self.losses = open(os.path.join(self.params.saving_path, "losses.txt"),"a")
            self.losses.write(f"{epoch}, {train_loss}, {val_loss}, {elapsed_time_sec}\n")
            self.losses.close()

            # Save the weights
            model_save_path = os.path.join(self.params.saving_path, "last.pth")
            torch.save(self.model.state_dict(), model_save_path)
            self.model.cuda()  # Move back to GPU
            
            # If this model is better then the previous one,
            # save this as the best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                model_save_path = os.path.join(self.params.saving_path, "best.pth")
                torch.save(self.model.state_dict(), model_save_path)
                self.model.cuda()  # Move back to GPU
            
            # Clean up additional memory after epoch
            torch.cuda.empty_cache()
            
if __name__ == "__main__":
    trainer = TrainingManager()
    trainer.train()