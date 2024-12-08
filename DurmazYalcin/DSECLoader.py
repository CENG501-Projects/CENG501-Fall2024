import torch
from torch.utils.data import Dataset, DataLoader
from DSECManipulator import DSECManipulator
import sys
import numpy as np
import cv2

import visualization_utils
from visualization_utils import txtAdder
seq_train = {
    0:"thun_00_a",
    1:"zurich_city_01_a",
    2:"zurich_city_02_a",
    3:"zurich_city_02_c",
    4:"zurich_city_02_d",
    5:"zurich_city_02_e",
    6:"zurich_city_03_a",
    7:"zurich_city_05_a",
    8:"zurich_city_05_b",
    9:"zurich_city_06_a",
    10:"zurich_city_07_a",
    11:"zurich_city_08_a",
    12:"zurich_city_09_a",
    13:"zurich_city_10_a",
    14:"zurich_city_10_b",
    15:"zurich_city_11_a",
    16:"zurich_city_11_b",
    17:"zurich_city_11_c"
}

seq_val = {
    0:"thun_00_a"
}

class DSECDataset(Dataset):
    def __init__(self, main_path, seq_dic):
        
        # Load Datasets
        self.data_sequences = {}
        self.index_upper_bounds   = {}
        
        data_length = 0
        for key in seq_dic:
            data   = DSECManipulator(main_path, seq_dic[key])
            data_length = data_length + data.flow_list.itemCount()
            
            self.data_sequences.update({key:data})
            self.index_upper_bounds.update({key:data_length})
        self.data_length = data_length
        
        self.items = self.getValidSeqIndexPairs()
        
        self.half_bins = 5
        
        self.draft_bin = np.zeros((4,data.height,data.width))
        
        self.txtManager = txtAdder()
        self.counter = 0
    
    def __len__(self):
        return self.data_length
    

    def __getitem__(self, idx):
        seq_key, local_idx = self.items[idx]
        
        events = self.data_sequences[seq_key].getEvents(local_idx)
        flow_gt = self.data_sequences[seq_key].getOF(local_idx)

        uv_gt = flow_gt[:,:,:2]
        uv_validity = flow_gt[:,:,2]
        
        time_interval = events['time_interval']
        
        t_start = time_interval[0]
        t_stop  = time_interval[2]
        bin_duration = float(t_stop-t_start) / float(2*self.half_bins)     
        
        input_tensors = []
        for bin_idx in range(self.half_bins):
            input_tensor = self.draft_bin.copy()
            #### First get the former group
            # Determine the window size
            win_start = (t_start + bin_idx * bin_duration) * 1e3
            win_stop  = (t_start + (bin_idx + 1) * bin_duration) * 1e3
            
            # Determine the roi fo the bin
            roi_1 = np.where((events['t'] > win_start) & (events['t'] <= win_stop) & (events['p'] == 1))[0]
            roi_0 = np.where((events['t'] > win_start) & (events['t'] <= win_stop) & (events['p'] == 0))[0]

            x_1 = events['x'][roi_1]
            y_1 = events['y'][roi_1]

            x_0 = events['x'][roi_0]
            y_0 = events['y'][roi_0]

            input_tensor[0,y_0,x_0] = 1
            input_tensor[1,y_1,x_1] = 1
            
            
            #### Second get the later group
            # Determine the window size
            win_start = (t_start + (bin_idx+self.half_bins) * bin_duration) * 1e3
            win_stop  = (t_start + (bin_idx+self.half_bins + 1) * bin_duration) * 1e3
            
            # Determine the roi fo the bin
            roi_1 = np.where((events['t'] > win_start) & (events['t'] <= win_stop) & (events['p'] == 1))[0]
            roi_0 = np.where((events['t'] > win_start) & (events['t'] <= win_stop) & (events['p'] == 0))[0]

            x_1 = events['x'][roi_1]
            y_1 = events['y'][roi_1]

            x_0 = events['x'][roi_0]
            y_0 = events['y'][roi_0]

            input_tensor[2,y_0,x_0] = 1
            input_tensor[3,y_1,x_1] = 1
            
            # Append the state
            input_tensors.append(input_tensor)

        item = {
            "input_tensors": input_tensors,
            "uv_gt" : uv_gt,
            "uv_validity": uv_validity
        }
            
        return item

    def visualize(self, idx):
        seq_key, local_idx = self.items[idx]

        # Get Events
        events = self.data_sequences[seq_key].getEvents(local_idx)
        
        # Get GT
        flow_gt = self.data_sequences[seq_key].getOF(local_idx)
        
        # Visualize All Events at Once
        t = events['t']
        time_interval = events['time_interval']
        
        t_start = time_interval[0]
        t_stop  = time_interval[2]
        bin_duration = float(t_stop-t_start) / float(2*self.half_bins)
        
        win_start = (t_start + 0  * bin_duration) * 1e3
        win_stop  = (t_start + 20 * bin_duration) * 1e3
        roi = np.where((t > win_start) & (t <= win_stop))[0]

        x = events['x'][roi]
        y = events['y'][roi]
        p = events['p'][roi]
        bin_img_all = visualization_utils.events_to_image(x,y,p,640,480)
        self.txtManager.put_txt(bin_img_all, "All Events For A Single OF Estimation")
        
        uv = flow_gt[:,:,:2]

        flow_vis = visualization_utils.flow_to_image(uv)
        self.txtManager.put_txt(flow_vis, "GT Optical Flow")

        
        for bin_idx in range(2*self.half_bins):
            win_start = (t_start + bin_idx * bin_duration) * 1e3
            win_stop  = (t_start + (bin_idx + 1) * bin_duration) * 1e3
            roi_1 = np.where((events['t'] > win_start) & (events['t'] <= win_stop) & (events['p'] == 1))[0]
            roi_0 = np.where((events['t'] > win_start) & (events['t'] <= win_stop) & (events['p'] == 0))[0]

            x_1 = events['x'][roi_1]
            y_1 = events['y'][roi_1]
            p_1 = events['p'][roi_1]
            bin_img_vis_1 = visualization_utils.events_to_image(x_1,y_1,p_1,640,480)
            self.txtManager.put_txt(bin_img_vis_1, "Binned ON Events")
            
            x_0 = events['x'][roi_0]
            y_0 = events['y'][roi_0]
            p_0 = events['p'][roi_0]
            bin_img_vis_0 = visualization_utils.events_to_image(x_0,y_0,p_0,640,480)
            
            self.txtManager.put_txt(bin_img_vis_0, "Binned OFF Events")
            
            out_img_up = cv2.hconcat([bin_img_all,flow_vis])
            out_img_down = cv2.hconcat([bin_img_vis_1,bin_img_vis_0])
            out_img = cv2.vconcat([out_img_up, out_img_down])
            cv2.imshow("out_img", out_img)
            cv2.imwrite("InputRepresentation/"+str(self.counter).zfill(3)+".png", out_img)
            self.counter = self.counter + 1
            cv2.waitKey(1)
            
        return None
    
    
    def getValidSeqIndexPairs(self):
        items = []
        for idx in range(self.data_length):
            seq_key, local_idx = self.global_idx_to_seq_and_local_idx(idx)
            items.append((seq_key,local_idx))

        return items
    
    def global_idx_to_seq_and_local_idx(self, global_idx):
        
        seq_key   = None
        local_idx = None
        
        lower_bound = 0
        for key in self.index_upper_bounds:
            if global_idx < self.index_upper_bounds[key]:
                seq_key   = key
                local_idx = global_idx - lower_bound
                break
            else:
                lower_bound = self.index_upper_bounds[key]
                
        if seq_key == None or local_idx == None:
            print("Something is wrong with dataloading")
            sys.exit()

        return seq_key, local_idx
    
        
if __name__ == "__main__":
    main_path   = "/home/hakito/datasets/event_data/dsec"
    
    train_data = DSECDataset(main_path, seq_train)
    print(len(train_data))

    valid_data = DSECDataset(main_path, seq_val)
    print(len(valid_data))
    
    # for idx in range(0,100):
    #     train_data.visualize(idx)
        
        
        
        