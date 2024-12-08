import h5py
import hdf5plugin
from numba import jit
import numpy as np
import os
import cv2
import imageio.v2 as imageio
import math
import torch

import utils
import visualization_utils


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
        self.flow_list = utils.itemList(self.flow_folder)
        
        # Flow timestamps
        self.flow_timestamps = np.loadtxt(flow_timestamps_file, comments="#", delimiter=",").reshape(-1,2)[:,0].reshape(-1).astype(int)
        
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
    
    
    def getCenterTime_us(self, idx):
        center_time_us = self.flow_timestamps[idx] - self.time_offset
        return center_time_us

    def getCenterTime_ms(self, idx):
        center_time_us = self.getCenterTime_us(idx)
        center_time_ms = int(center_time_us * (1e-3))
        return center_time_ms
    
    def getEvents(self, idx):
        
        center_time_ms = self.getCenterTime_ms(idx)
        t_start_ms = center_time_ms - self.win_size_ms
        t_end_ms   = center_time_ms + self.win_size_ms

        time_interval = (t_start_ms, center_time_ms, t_end_ms)
        
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx   = self.ms2idx(t_end_ms)
        
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
        
    def getEventImage(self,idx):
        t_start_us = self.flow_timestamps[idx] - self.time_offset
        t_end_us   = self.flow_timestamps[idx+1] - self.time_offset
        
        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx   = self.ms2idx(t_end_ms)
        
        t = self.events['t'][t_start_ms_idx : t_end_ms_idx]
        x = self.events['x'][t_start_ms_idx : t_end_ms_idx]
        y = self.events['y'][t_start_ms_idx : t_end_ms_idx]
        p = self.events['p'][t_start_ms_idx : t_end_ms_idx]
    
        points_on_background = visualization_utils.events_to_image(x,y,p,self.width,self.height)
        return points_on_background
    
    def getRectifiedEventImage(self,idx):
        t_start_us = self.flow_timestamps[idx] - self.time_offset
        t_end_us   = self.flow_timestamps[idx+1] - self.time_offset
        
        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx   = self.ms2idx(t_end_ms)
        
        t = self.events['t'][t_start_ms_idx : t_end_ms_idx]
        x = self.events['x'][t_start_ms_idx : t_end_ms_idx]
        y = self.events['y'][t_start_ms_idx : t_end_ms_idx]
        p = self.events['p'][t_start_ms_idx : t_end_ms_idx]
        
        rectified_coordinates = self.rect_map[y,x]
        x_rectified = rectified_coordinates[..., 0]
        y_rectified = rectified_coordinates[..., 1]
    
        points_on_background = visualization_utils.events_to_image(x_rectified,y_rectified,p,self.width,self.height)
        return points_on_background
    
    
    def getEventImage2(self,t_start_us, t_end_us):       
        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx   = self.ms2idx(t_end_ms)

        t = self.events['t'][t_start_ms_idx : t_end_ms_idx]
        x = self.events['x'][t_start_ms_idx : t_end_ms_idx]
        y = self.events['y'][t_start_ms_idx : t_end_ms_idx]
        p = self.events['p'][t_start_ms_idx : t_end_ms_idx]
        
        points_on_background = visualization_utils.events_to_image(x,y,p,self.width,self.height)
        return points_on_background

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
    main_path   = "/home/hakito/datasets/event_data/dsec"
    seq         = "zurich_city_09_a"
    data        = DSECManipulator(main_path, seq)


    for idx in range(0,data.flow_list.itemCount()-1):
        event_image = data.getEventImage(idx)
        rectified_event_image = data.getRectifiedEventImage(idx)
        
        of = data.getOF(idx)
        
        uv = of[:,:,:2]
        uv_validity = of[:,:,2]
        
        flow_vis = visualization_utils.flow_to_image(uv)
        
        out_image = cv2.hconcat([event_image,rectified_event_image])
        cv2.imshow("out_image", out_image)
        cv2.imshow("flow_vis", flow_vis)
        
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
