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
from utils.visualization_utils import events_to_event_image, flow_to_image

class DSECManipulator(Dataset):
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
        self.draft_bin = np.zeros((4,480,640))
        self.height = 480
        self.width  = 640
    
    def __getitem__(self, idx):
        folder, local_idx   = self.dataAddresses[idx]
        data_path           = self.datasets[folder].getItemPath(local_idx)
        data                = h5py.File(str(data_path), 'r')
        
        x        = np.array(data['x'])
        y        = np.array(data['y'])
        t        = np.array(data['t'])
        p        = np.array(data['p'])
        
        left_events = np.zeros((x.shape[0],4))
        left_events[:,0] = x.reshape(-1)
        left_events[:,1] = y.reshape(-1)
        left_events[:,2] = t.reshape(-1)
        left_events[:,3] = p.reshape(-1)
        
        img_times           = np.array(data['img_times'])
        t_start = img_times[0]
        t_mid   = img_times[1]
        t_stop  = img_times[2]
        
        
        optical_flow = np.array(data["optical_flow"])

        left_event_tensor  = self.event_bins(events=left_events, t_start=t_start, t_mid=t_mid, t_stop=t_stop, cam="left")

        item = {
            "left_events": left_event_tensor,
            "optical_flow": optical_flow.transpose(2,0,1)
        }

        return item

    def __len__(self):
        return self.data_count
    
    def event_bins(self, events, t_start, t_mid,  t_stop, cam="cam"):
        input_tensors = []
        
        # First get the former events
        bin_duration_former = float(t_mid-t_start) / float(self.half_bins)   
        bin_duration_later  = float(t_stop-t_mid) / float(self.half_bins)   
        
        if self.visualizeBins:
            eventFrame = events_to_event_image(events, height=self.height, width=self.width)

        for bin_idx in range(self.half_bins):
            # Get the draft of our event representation
            input_tensor = self.draft_bin.copy()
            
            #### First get the former group
            # Determine the window size
            win_start = (t_start + bin_idx * bin_duration_former)
            win_stop  = (t_start + (bin_idx + 1) * bin_duration_former)
            
            # Determine the roi for the bin
            roi_1 = np.where((events[:,2] > win_start) & (events[:,2] <= win_stop) & (events[:,3] == 1))[0]
            roi_0 = np.where((events[:,2] > win_start) & (events[:,2] <= win_stop) & (events[:,3] == 0))[0]
            
            if self.visualizeBins:
                eventFrameFormerP = events_to_event_image(events[roi_1], height=self.height, width=self.width)
                eventFrameFormerN = events_to_event_image(events[roi_0], height=self.height, width=self.width)

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
            roi_0 = np.where((events[:,2] > win_start) & (events[:,2] <= win_stop) & (events[:,3] == 0))[0]

            if self.visualizeBins:
                eventFrameLaterP = events_to_event_image(events[roi_1], height=self.height, width=self.width)
                eventFrameLaterN = events_to_event_image(events[roi_0], height=self.height, width=self.width)
                
                eventBins = cv2.hconcat([eventFrame, eventFrameFormerP, eventFrameFormerN, eventFrameLaterP, eventFrameLaterN])
                cv2.imshow(cam + " eventBins", eventBins)
                cv2.waitKey(1)
                
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
        return input_tensors
    
if __name__ == "__main__":
    main_path = "/media/hakito/HDD/event_data/dsec/ProcessedDSEC"
    DSECdata = DSECManipulator(main_path)
    DSECdata.visualizeBins = True
    print(len(DSECdata))
    for item in DSECdata:
        of = item["optical_flow"]
        
        uv = of[:,:,:2]
        uv_validity = of[:,:,2]
        
        flow_vis = flow_to_image(uv)
        cv2.imshow("flow_vis", flow_vis)
        
        key = cv2.waitKey(1)
        if key == ord("q"):
            break