# Generale libraries that can be installed with pip
import os, shutil, sys
from torch.utils.data import Dataset, DataLoader
import time
import h5py
import hdf5plugin
import numpy as np
import cv2

# Local Libraries
from utils.utils import itemList
from utils.visualization_utils import events_to_event_image

class MVSECManipulator(Dataset):
    def __init__(self,main_path):
        self.main_path = main_path
        
        # Get a list of available dataset
        self.folders = [folder for folder in os.listdir(self.main_path) if os.path.isdir(os.path.join(self.main_path, folder))]
        self.folders.sort()
        
        self.datasets      = {}
        self.dataAddresses = {}
        
        self.visualizeBins = False
                
        self.data_count = 0
        for folder in self.folders:
            path = os.path.join(self.main_path,folder)
            dataList = itemList(path)
            self.datasets.update({folder:dataList})
            
            
            # Create an index for dataset
            new_data_count = dataList.itemCount()
            for local_idx in range(new_data_count):
                self.dataAddresses.update({self.data_count: [folder, local_idx]})
                self.data_count += 1
    
        self.half_bins = 5
        # Let us determine the width and height 
        self.draft_bin = np.zeros((4,260,346))
    
    
    def __getitem__(self, idx):
        folder, local_idx   = self.dataAddresses[idx]
        data_path           = self.datasets[folder].getItemPath(local_idx)
        data                = h5py.File(str(data_path), 'r')
        left_events         = np.ascontiguousarray(data['left_events'])
        right_events        = np.ascontiguousarray(data['right_events'])
        curr_raw_img        = np.ascontiguousarray(data['curr_raw_img'])
        next_raw_img        = np.ascontiguousarray(data['next_raw_img'])
        
        img_times           = np.array(data['img_times'])
        t_start = img_times[0]
        t_mid   = img_times[1]
        t_stop  = img_times[2]
        
        left_event_tensor  = self.event_bins(events=left_events, t_start=t_start, t_mid=t_mid, t_stop=t_stop, cam="left")
        right_event_tensor = self.event_bins(events=right_events, t_start=t_start, t_mid=t_mid, t_stop=t_stop, cam="right")
        
        data_dic = {
            'left_events'       : left_event_tensor,
            'right_events'      : right_event_tensor,
            'curr_raw_img'      : curr_raw_img,
            'next_raw_img'      : next_raw_img,
        }
        
        return data_dic

    def __len__(self):
        return self.data_count
    
    def event_bins(self, events, t_start, t_mid,  t_stop, cam="cam"):

        input_tensors = []
        
        # First get the former events
        bin_duration_former = float(t_mid-t_start) / float(self.half_bins)   
        bin_duration_later  = float(t_stop-t_mid) / float(self.half_bins)   
        
        if self.visualizeBins:
            eventFrame = events_to_event_image(events)

        for bin_idx in range(self.half_bins):
            # Get the draft of our event representation
            input_tensor = self.draft_bin.copy()
            
            #### First get the former group
            # Determine the window size
            win_start = (t_start + bin_idx * bin_duration_former)
            win_stop  = (t_start + (bin_idx + 1) * bin_duration_former)
            
            # Determine the roi for the bin
            roi_1 = np.where((events[:,2] > win_start) & (events[:,2] <= win_stop) & (events[:,3] == 1))[0]
            roi_0 = np.where((events[:,2] > win_start) & (events[:,2] <= win_stop) & (events[:,3] == -1))[0]
            
            if self.visualizeBins:
                eventFrameFormerP = events_to_event_image(events[roi_1])
                eventFrameFormerN = events_to_event_image(events[roi_0])
            
            ## Get the xy coordinates of the ON and OFF events
            # ON Events
            x_1 = events[roi_1,0].astype(int)
            y_1 = events[roi_1,1].astype(int)
            # OFF Events
            x_0 = events[roi_0,0].astype(int)
            y_0 = events[roi_0,1].astype(int)
            
            # Fill the tensor
            input_tensor[0,y_0,x_0] = 1
            input_tensor[1,y_1,x_1] = 1
            
            #### Now get the later group
            # Determine the window size
            win_start = (t_mid + bin_idx * bin_duration_later)
            win_stop  = (t_mid + (bin_idx + 1) * bin_duration_later)
            
            # Determine the roi for the bin
            roi_1 = np.where((events[:,2] > win_start) & (events[:,2] <= win_stop) & (events[:,3] == 1))[0]
            roi_0 = np.where((events[:,2] > win_start) & (events[:,2] <= win_stop) & (events[:,3] == -1))[0]
            
            if self.visualizeBins:
                
                eventFrameLaterP = events_to_event_image(events[roi_1])
                eventFrameLaterN = events_to_event_image(events[roi_0])
                
                eventBins = cv2.hconcat([eventFrame, eventFrameFormerP, eventFrameFormerN, eventFrameLaterP, eventFrameLaterN])
                cv2.imshow(cam + " eventBins", eventBins)
                
            ## Get the xy coordinates of the ON and OFF events
            # ON Events
            x_1 = events[roi_1,0].astype(int)
            y_1 = events[roi_1,1].astype(int)
            # OFF Events
            x_0 = events[roi_0,0].astype(int)
            y_0 = events[roi_0,1].astype(int)
            
            # Fill the tensor
            input_tensor[2,y_0,x_0] = 1
            input_tensor[3,y_1,x_1] = 1
            # Append the state
            input_tensors.append(input_tensor)
            
            if self.visualizeBins:
                cv2.waitKey(1)
        return input_tensors
    
    
    
if __name__ == "__main__":
    mvsec_path = "/media/hakito/HDD/event_data/mvsec/OpticalFlow"
    
    dataMVSEC  = MVSECManipulator(mvsec_path)
    dataMVSEC.visualizeBins = False
    dataLoader = DataLoader(dataMVSEC,batch_size=1, shuffle=False)
    
    for idx, item in enumerate(dataLoader):
        left_events = item['left_events']
        right_events = item['right_events']
        curr_raw_img = item['curr_raw_img']
        next_raw_img = item['next_raw_img']

        key = cv2.waitKey(1)    
        if key == ord("q"):
            break