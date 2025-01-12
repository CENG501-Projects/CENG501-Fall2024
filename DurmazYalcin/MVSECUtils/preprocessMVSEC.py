import numpy as np
import os, shutil
import h5py
import time
import cv2
import torch
import sys

import matplotlib.pyplot as plt
plt.rc('mathtext', fontset='cm')  # Use Computer Modern (LaTeX default) fonts
plt.rc('font', family='serif')   # Use serif fonts for text

def visualze_dt(data):
    data = data.reshape(-1)
    dt_arr = np.zeros((data.shape[0]-1))
    dt_arr = data[1:] - data[:-1]
    
    plt.plot(data[:-1]-data[0], dt_arr)
    plt.xlabel("Time")
    plt.xlabel(r"$dt$")
    plt.show()
    
class RawMVSECManipulator:
    def __init__(self, main_path, sequence):
        
        self.main_path = main_path
        self.sequence  = sequence
        
        # Path to data and gt
        path_to_data = os.path.join(self.main_path, sequence + "_data.hdf5")
        path_to_gt   = os.path.join(self.main_path, sequence + "_gt.hdf5")

        # Load the data
        self.hdf_data = h5py.File(path_to_data, "r")
        self.hdf_gt = h5py.File(path_to_gt, "r")
        
        # Get the events, x, y, t, p
        self.left_event_data  = self.hdf_data['davis']['left']['events']
        self.right_event_data = self.hdf_data['davis']['right']['events']
        
        # Get the image frames
        self.left_image_data  = self.hdf_data['davis']['left']['image_raw']
        self.left_image_ts    = self.hdf_data['davis']['left']['image_raw_ts']
        
        # Load rectified depth
        self.rectified_depth    = self.hdf_gt['davis']['left']['depth_image_rect']
        self.rectified_depth_ts = self.hdf_gt['davis']['left']['depth_image_rect_ts']
   
        # Load optical flows
        self.optical_flows   = self.hdf_gt['davis']['left']['flow_dist']
        self.optical_flow_ts = self.hdf_gt['davis']['left']['flow_dist_ts']
        
        # Load groundtruth pose
        self.left_poses   = self.hdf_gt['davis']['left']['odometry']
        self.left_pose_ts = self.hdf_gt['davis']['left']['odometry_ts']
        
        # self.calculate_event_index()
        
    def calculate_event_index(self):
        left_event_times = self.left_event_data[:,2]
        left_event_times = np.array(left_event_times)
        
        right_event_times = self.right_event_data[:,2]
        right_event_times = np.array(right_event_times)
        
        self.left_event_indexes = []
        self.right_event_indexes = []
        
        for idx in range(self.rectified_depth_ts.shape[0]-1):
            print(f"{idx} / {self.rectified_depth_ts.shape[0]-1}")
            # Get the time limits
            curr_depth_time = self.rectified_depth_ts[idx]
            next_depth_time = self.rectified_depth_ts[idx+1]
            
            # Event indexes of the left 
            left_indexes = np.where(
                (left_event_times>= curr_depth_time) &
                (left_event_times < next_depth_time)
            )
            left_indexes = (left_indexes[0][0], left_indexes[0][-1])
            self.left_event_indexes.append(left_indexes)
            
            # Event indexes of the right
            right_indexes = np.where(
                (right_event_times>= curr_depth_time) &
                (right_event_times < next_depth_time)
            )
            right_indexes = (right_indexes[0][0], right_indexes[0][-1])
            self.right_event_indexes.append(right_indexes)
        
        
if __name__ == "__main__":
    
    sequences = ["indoor_flying1", "indoor_flying2", "indoor_flying3", "indoor_flying4", "outdoor_night1", "outdoor_night2", "outdoor_night3", "outdoor_day1", "outdoor_day2"]
    
    # Main path
    main_path = "/media/hakito/HDD/event_data/mvsec"
    
    for sequence in sequences:
        # Create the saving path
        saving_path = os.path.join(main_path,"OpticalFlow",sequence)
        if os.path.exists(saving_path):
            shutil.rmtree(saving_path)
        os.makedirs(saving_path)
        
        # Initialize the MVSEC manipulator
        dataMVSEC = RawMVSECManipulator(main_path=main_path, sequence=sequence)
        left_event_times = dataMVSEC.left_event_data[:,2]
        left_event_times = np.array(left_event_times)
        
        right_event_times = dataMVSEC.right_event_data[:,2]
        right_event_times = np.array(right_event_times)
            
        
        for frame_idx in range(1,dataMVSEC.left_image_ts.shape[0]-2):
            print(f"{frame_idx} / {dataMVSEC.left_image_ts.shape[0]-2}")
            prev_time = dataMVSEC.left_image_ts[frame_idx-1]
            curr_time = dataMVSEC.left_image_ts[frame_idx]
            next_time = dataMVSEC.left_image_ts[frame_idx+1]
            
            # Event indexes of the left 
            left_indexes = np.where(
                (left_event_times>= prev_time) &
                (left_event_times < next_time)
            )
            left_indexes = (left_indexes[0][0], left_indexes[0][-1])

            # Event indexes of the right
            right_indexes = np.where(
                (right_event_times>= prev_time) &
                (right_event_times < next_time)
            )
            right_indexes = (right_indexes[0][0], right_indexes[0][-1])

            # Get the related events
            left_events = dataMVSEC.left_event_data[left_indexes[0]:left_indexes[1]]
            right_events = dataMVSEC.right_event_data[right_indexes[0]:right_indexes[1]]
        
            # Get the frames
            prev_img  = dataMVSEC.left_image_data[frame_idx-1, :, :]
            curr_img  = dataMVSEC.left_image_data[frame_idx, :, :]
            next_img  = dataMVSEC.left_image_data[frame_idx+1, :, :]

            img_times = np.array([np.array(dataMVSEC.left_image_ts[frame_idx-1]), np.array(dataMVSEC.left_image_ts[frame_idx]), np.array(dataMVSEC.left_image_ts[frame_idx+1])])
            # Save the processed data
            out_database_path = os.path.join(saving_path, str(frame_idx).zfill(5) + ".hdf5")
            with h5py.File(out_database_path, 'w') as out_database:
                out_database.create_dataset('left_events', data=left_events)
                out_database.create_dataset('right_events', data=right_events)
                out_database.create_dataset('prev_raw_img', data=prev_img)
                out_database.create_dataset('curr_raw_img', data=curr_img)
                out_database.create_dataset('next_raw_img', data=next_img)
                out_database.create_dataset('img_times', data=img_times)

        
