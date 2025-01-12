import h5py
import hdf5plugin

import numpy as np
import os, shutil, sys
import cv2
import math


class itemList:
    def __init__(self, path) -> None:
        self.path = path
        folder_items = os.listdir(path)
        folder_items.sort()
        first_item = folder_items[0]
        second_item = folder_items[1]
        self.extension = first_item.split(".")[-1]
        self.extension_length = len(self.extension) + 1
        
        # Check if we have zero padding in the namming
        self.zero_padding = False
        self.padding_length = 0
        if first_item[0] == "0" and second_item[0] == "0": 
            self.zero_padding = True
            self.padding_length = len(first_item[:-self.extension_length])

        self.item_ids = []
        for item in folder_items:
            try:
                item_id = int(item[:-self.extension_length])
                self.item_ids.append(item_id)
            except:
                print(f"Item ({item[:-self.extension_length]}) cannot be converted to int. It will be discarded.")

        self.item_ids.sort()
        
    def getItemPath(self,idx:int):
        if self.zero_padding:
            path = os.path.join(self.path, str(self.item_ids[idx]).zfill(self.padding_length)+"." + self.extension)
        else:
            path = os.path.join(self.path, str(self.item_ids[idx])+"." + self.extension)
        return path
    
    def getItemPathFromName(self,name):
        path = os.path.join(self.path, name + "." + self.extension)
        return path
    
    def getItemID(self,idx):
        return self.item_ids[idx]
    
    def getItemName(self,idx):
        if self.zero_padding:
            name = str(self.item_ids[idx]).zfill(self.padding_length)
        else:
            name = str(self.item_ids[idx])
        return name

    def itemCount(self):
        return len(self.item_ids)


class DSECManipulator:
    def __init__(self, main_path, seq):
        # Path to event data
        self.data_path = os.path.join(main_path,seq,seq+"_events_left","events.h5")
        rect_map_path = os.path.join(main_path,seq,seq+"_events_left","rectify_map.h5")
        self.flow_folder = os.path.join(main_path,seq,seq+"_optical_flow_forward_event")
        flow_timestamps_file = os.path.join(main_path,seq,seq+"_optical_flow_forward_timestamps.txt")
        
        # Read the data
        self.h5f_events = h5py.File(str(self.data_path), 'r')
        self.h5f_rect_map = h5py.File(str(rect_map_path), 'r')
        
        # Get the events
        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f_events['events/{}'.format(dset_str)]
        # Get time to event index map
        self.ms_to_idx = np.asarray(self.h5f_events['ms_to_idx'], dtype='int64')
        
        # Get the time offset
        self.time_offset = self.h5f_events['t_offset'][()]
        
        # Window Size ms
        self.win_size_ms = 100

        # Get rectification map
        self.rect_map = np.asarray(self.h5f_rect_map['rectify_map'][()], dtype='int64')
        
        # Get the list of optical flows
        self.flow_list = itemList(self.flow_folder)
        
        # Flow timestamps
        self.flow_timestamps = np.loadtxt(flow_timestamps_file, comments="#", delimiter=",").reshape(-1,2).astype(int)
        
        # Varşfy that we have a timestamp for each flow
        assert self.flow_list.itemCount() == self.flow_timestamps.shape[0]
        
        # Create an empty draft for optical flow
        self.width  = 640
        self.height = 480
        
    def getOF(self, idx):
        path_to_OF = self.flow_list.getItemPath(idx)
        OF = cv2.imread(path_to_OF, cv2.IMREAD_UNCHANGED)
        OF = cv2.cvtColor(OF, cv2.COLOR_BGR2RGB).astype(float)
        OF[:,:,:2] = (OF[:,:,:2] - (2**15)) / (128.0)
        return OF
    
    def getStartStop_us(self, idx):
        start_time_us = self.flow_timestamps[idx,0] - self.time_offset
        stop_time_us  = self.flow_timestamps[idx,1] - self.time_offset
        return start_time_us, stop_time_us

    def getStartStop_ms(self, idx):
        start_time_us, stop_time_us = self.getStartStop_us(idx)
        start_time_ms, stop_time_ms = int(start_time_us * (1e-3)), int(stop_time_us * (1e-3))
        return start_time_ms, stop_time_ms
    
    
    def getEvents(self, idx):  
        # Get the start and stop indexes of the current window      
        t_start_ms, t_end_ms = self.getStartStop_ms(idx)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx   = self.ms2idx(t_end_ms)
        
        # Get the timing information of the images
        t_start_us, t_end_us = self.getStartStop_us(idx)
        center_time_us = int(0.5*(t_start_us + t_end_us))
        time_interval = (t_start_us, center_time_us, t_end_us)
        
        t = self.events['t'][t_start_ms_idx : t_end_ms_idx]
        x = self.events['x'][t_start_ms_idx : t_end_ms_idx]
        y = self.events['y'][t_start_ms_idx : t_end_ms_idx]
        p = self.events['p'][t_start_ms_idx : t_end_ms_idx]
        
        events = {
            't': t,
            'x': x,
            'y': y,
            'p': p,
            'time_interval': time_interval
        }
        
        return events
        

    def get_conservative_window_ms(self, ts_start_us: int, ts_end_us) -> tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms
    
    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]   
    

if __name__ == "__main__":
    main_path   = "Sequences"
    
    sequences = ["thun_00_a", 
                 "zurich_city_01_a", 
                 "zurich_city_02_a", "zurich_city_02_c", "zurich_city_02_d", "zurich_city_02_e", 
                 "zurich_city_03_a",
                 "zurich_city_05_a", "zurich_city_05_b", 
                 "zurich_city_06_a", 
                 "zurich_city_07_a", 
                 "zurich_city_08_a", 
                 "zurich_city_09_a", 
                 "zurich_city_10_a", "zurich_city_10_b", 
                 "zurich_city_11_a", "zurich_city_11_b", "zurich_city_11_c"]
    
    sequences = ["zurich_city_10_a", "zurich_city_10_b", 
                 "zurich_city_11_a", "zurich_city_11_b", "zurich_city_11_c"]
    
    for seq in sequences:
        data        = DSECManipulator(main_path, seq)
        
        # Create the saving path
        saving_path = os.path.join("ProcessedDSEC",seq)
        if os.path.exists(saving_path):
            shutil.rmtree(saving_path)
        os.makedirs(saving_path)
        
        for frame_idx in range(0,data.flow_list.itemCount()-1):
            events = data.getEvents(frame_idx)
            gt     = data.getOF(frame_idx)
            img_times = events['time_interval']
            # Save the processed data
            out_database_path = os.path.join(saving_path, str(frame_idx).zfill(5) + ".hdf5")
            with h5py.File(out_database_path, 'w') as out_database:
                out_database.create_dataset('x', data=events["x"])
                out_database.create_dataset('y', data=events["y"])
                out_database.create_dataset('t', data=events["t"])
                out_database.create_dataset('p', data=events["p"])
                out_database.create_dataset('optical_flow', data=gt)
                out_database.create_dataset('img_times',   data=img_times)