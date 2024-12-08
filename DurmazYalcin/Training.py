import torch

from SNN import SpikeFlow
from DSECLoader import DSECDataset, seq_train, seq_val
import cv2
import numpy as np


import time
import sys


class parameters:
    def __init__(self):
        self.epochs       = 30
        
        self.batch_size   = 8
        
        self.lr           = 5e-5 
        self.momentum     = 0.9
        self.weight_decay = 4e-4
        self.beta         = 0.999
        

def get_scaled_validty_tensors(validity: np.array):
    validity_tensor_arr = []
    old_height = validity.shape[0]
    old_width  = validity.shape[1]

    for idx in range(4):
        new_height = int(old_height / (2**idx))
        new_width  = int(old_width / (2**idx))
        validty_tensor = torch.from_numpy(cv2.resize(validity, (new_width,new_height), interpolation=cv2.INTER_LINEAR)).float().cuda().unsqueeze(0).unsqueeze(0)
        validty_tensor = torch.cat([validty_tensor, validty_tensor], dim=1)
        validity_tensor_arr.append(validty_tensor)
    return validity_tensor_arr
    
def get_scaled_flows(flow:np.array):
    flow_tensor_arr = []
    old_height = flow.shape[0]
    old_width  = flow.shape[1]

    for idx in range(4):
        new_height = int(old_height / (2**idx))
        new_width  = int(old_width / (2**idx))
        
        scaled_flow = cv2.resize(flow, (new_width,new_height), interpolation=cv2.INTER_LINEAR) / (2**idx)
        
        scaled_u = scaled_flow[:,:,0] 
        scaled_u_tensor = torch.from_numpy(scaled_u).float().cuda().unsqueeze(0).unsqueeze(0)
        
        scaled_v = scaled_flow[:,:,1] 
        scaled_v_tensor = torch.from_numpy(scaled_v).float().cuda().unsqueeze(0).unsqueeze(0)
        
        scaled_flow_tensor = torch.cat([scaled_u_tensor, scaled_v_tensor], dim=1).float().cuda()
        flow_tensor_arr.append(scaled_flow_tensor)
        
    return flow_tensor_arr

def computeLoss(uv_gt, uv_validty, estimated_scaled_flows):
    # Create scaled masks and groundtruth
    validity_tensors    = get_scaled_validty_tensors(uv_validty)
    scaled_flow_tensors = get_scaled_flows(uv_gt)
    
    loss = 0.0
    for idx in range(4):
        estimated_flow  = estimated_scaled_flows[idx]
        flow_gt         = scaled_flow_tensors[idx]
        mask            = validity_tensors[idx]

        discrepeancy = (flow_gt - estimated_flow)
        masked_discrepeancy = torch.mul(discrepeancy,mask)
        loss = loss + torch.mul(masked_discrepeancy,masked_discrepeancy).sum() / mask.sum()

    return loss

        
class TrainingManager:
    def __init__(self):
        self.model      = SpikeFlow().cuda()
        self.params     = parameters()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.params.lr, betas=(self.params.momentum, self.params.beta))
        
        main_path   = "/home/hakito/datasets/event_data/dsec"
        self.train_data = DSECDataset(main_path, seq_train)
        self.valid_data = DSECDataset(main_path, seq_val)
        
        self.losses = open("losses.txt","w")
        self.losses.write("# epoch, training loss, validation loss, epoch duration (sec)")
        
        print(f"Len of training data : {len(self.train_data)}")
        print(f"Len of validation data : {len(self.valid_data)}")

    def train(self):
        best_val_loss = 1e10
        for epoch in range(self.params.epochs):
            t_start = time.monotonic_ns()
            ################# Training #################
            train_loss = 0.0
            counter = 0
            for item in self.train_data:
                input_representation = item['input_tensors']
                torch_tensors = [torch.from_numpy(input) for input in input_representation]
                torch_tensors = [t.unsqueeze(0) for t in torch_tensors]
                input = torch.cat(torch_tensors, dim=1).float().cuda()

                
                estimated_scaled_flows = self.model(input)
                
                loss = computeLoss(item['uv_gt'],item['uv_validity'],estimated_scaled_flows)
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                counter += 1


            train_loss /= len(self.train_data)
            
            ################# Validation #################
            val_loss = 0.0
            for item in self.valid_data:
                input_representation = item['input_tensors']
                torch_tensors = [torch.from_numpy(input) for input in input_representation]
                torch_tensors = [t.unsqueeze(0) for t in torch_tensors]
                input = torch.cat(torch_tensors, dim=1).float().cuda()
                
                estimated_scaled_flows = self.model(input)
                
                loss = computeLoss(item['uv_gt'],item['uv_validity'], estimated_scaled_flows)
                val_loss += loss.item()

            val_loss /= len(self.valid_data)
            
            t_stop = time.monotonic_ns()
            elapsed_time_sec = (t_stop - t_start) * (1e-9)
            print(f"Training Loss: {train_loss}, Validation Loss: {val_loss}. Elapsed Time : {elapsed_time_sec} sec. Data Count:{counter}")
            self.losses.write(f"{epoch}, {train_loss}, {val_loss}, {elapsed_time_sec}\n")
        
            model_save_path = "checkpoints/last.pth"
            torch.save(self.model.state_dict(), model_save_path)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                model_save_path = "checkpoints/best.pth"
                torch.save(self.model.state_dict(), model_save_path)
                
if __name__ == "__main__":
    trainer = TrainingManager()
    trainer.train()