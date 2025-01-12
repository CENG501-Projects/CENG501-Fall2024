from MVSECUtils.MVSECLoader import MVSECManipulator
from models.EventFlow import EventFlow
from utils.visualization_utils import flow_to_image
from utils.losses import multi_scale_photometric_loss

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os, shutil, time, sys

class parameters:
    def __init__(self):
        self.train_data_path    = "/media/ogam-mgeo/acikdata/Additional/OpticalFlow/Training"
        self.valid_data_path    = "/media/ogam-mgeo/acikdata/Additional/OpticalFlow/Validation"

        self.saving_path  = "checkpoints/MVSEC/ConvNet" 
        self.epochs       = 50
        
        self.batch_size   = 128
        
        self.lr           = 1e-4 
        self.momentum     = 0.9
        self.weight_decay = 4e-4
        self.beta         = 0.999
        
        
class trainManager:
    def __init__(self):
        # Create the saving path
        self.params = parameters()
        if not os.path.exists(self.params.saving_path):
            os.makedirs(self.params.saving_path)
        
        # Initialize the model
        self.model      = EventFlow().cuda()
        
        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.params.lr, betas=(self.params.momentum, self.params.beta))
        
        # Create data loader
        self.train_data_path = self.params.train_data_path
        self.valid_data_path = self.params.valid_data_path

        trainDataMVSEC  = MVSECManipulator(self.train_data_path)
        self.train_data = DataLoader(trainDataMVSEC,batch_size=self.params.batch_size, shuffle=False)

        validDataMVSEC  = MVSECManipulator(self.valid_data_path)
        self.valid_data = DataLoader(validDataMVSEC,batch_size=self.params.batch_size, shuffle=False)

        # Save the losses as well
        self.losses = open(os.path.join(self.params.saving_path, "losses.txt"),"w")
        self.losses.write("# epoch, training loss, validation loss, epoch duration (sec)\n")
        self.losses.close()
        
    def train(self):
        best_loss = 1e10
        for epoch in range(self.params.epochs):
            t_start = time.monotonic_ns()

            ################# Training #################
            train_loss = 0.0
            for idx, item in enumerate(self.train_data):
                # Get events
                binarized_events = item['left_events']
                
                # Prepare the input
                input = torch.cat(binarized_events, dim=1).float().cuda()
                
                # Estimate the optical flow
                scaled_flows   = self.model(input)
                
                # Get image frames
                curr_image_frame = item['curr_raw_img'].unsqueeze(1)
                next_image_frame = item['next_raw_img'].unsqueeze(1)
                
                photometric_loss = multi_scale_photometric_loss(scaled_flows=scaled_flows, curr_frame=curr_image_frame, next_frame=next_image_frame)

                self.optimizer.zero_grad()
                photometric_loss.backward()
                self.optimizer.step()
                
                train_loss += photometric_loss.item()

                # Clean up memory
                del input, scaled_flows, photometric_loss, curr_image_frame, next_image_frame   # Free tensors
                torch.cuda.empty_cache()
            train_loss /= len(self.train_data)

            ################# Validation #################
            valid_loss = 0.0
            with torch.no_grad():
                for idx, item in enumerate(self.valid_data):
                    # Get events
                    binarized_events = item['left_events']
                    
                    # Prepare the input
                    input = torch.cat(binarized_events, dim=1).float().cuda()
                    
                    # Estimate the optical flow
                    scaled_flows   = self.model(input)
                    
                    # Get image frames
                    curr_image_frame = item['curr_raw_img'].unsqueeze(1)
                    next_image_frame = item['next_raw_img'].unsqueeze(1)
                    
                    photometric_loss = multi_scale_photometric_loss(scaled_flows=scaled_flows, curr_frame=curr_image_frame, next_frame=next_image_frame)

                    valid_loss += photometric_loss.item()

                    # Clean up memory
                    del input, scaled_flows, photometric_loss, curr_image_frame, next_image_frame   # Free tensors
                    torch.cuda.empty_cache()

            valid_loss /= len(self.valid_data)

            # Print the epoch
            t_stop = time.monotonic_ns()
            elapsed_time_sec = (t_stop - t_start) * (1e-9)
            print(f"Epoch:{epoch}/{self.params.epochs} -- Training Loss: {train_loss}. Validation Loss: {valid_loss}. Elapsed Time : {elapsed_time_sec} sec")
            
            self.losses = open(os.path.join(self.params.saving_path, "losses.txt"),"a")
            self.losses.write(f"{epoch}, {train_loss}, {valid_loss}, {elapsed_time_sec}\n")
            self.losses.close()

            # Save the weights
            model_save_path = os.path.join(self.params.saving_path, "last.pth")
            torch.save(self.model.state_dict(), model_save_path)
            self.model.cuda()  # Move back to GPU
            
            # If this model is better then the previous one,
            # save this as the best
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                model_save_path = os.path.join(self.params.saving_path, "best.pth")
                torch.save(self.model.state_dict(), model_save_path)
                self.model.cuda()  # Move back to GPU
            
            # Clean up additional memory after epoch
            torch.cuda.empty_cache()
            
if __name__ == "__main__":
    trainer = trainManager()
    trainer.train()
    