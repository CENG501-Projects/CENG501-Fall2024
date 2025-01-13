import torch
import cv2
import numpy as np

from models.EventFlow import EventFlow
from DSECUtils.DSECLoader import DSECManipulator

from torch.utils.data import DataLoader

from utils.visualization_utils import txtAdder
from utils import visualization_utils
import time
import sys
from utils.utils import padding

    
if __name__ == "__main__":
    model      = EventFlow().cuda()
    model.load_state_dict(torch.load('checkpoints/DSEC/ConNet/best.pth', weights_only=True))
    
    path_to_data = "/media/hakito/HDD/event_data/dsec/ProcessedDSEC/validation"
    
    data = DataLoader( DSECManipulator(path_to_data) )
    
    txtManager = txtAdder()
    with torch.no_grad():
        for idx,item in enumerate(data):
            # Get events
            binarized_events = item['left_events']

            # Prepare the input
            input = torch.cat(binarized_events, dim=1).float().cuda()
            
            # Estimate the optical flow
            estimated_scaled_flows   = model(input)
            
            # We are only interested in the not-scaled flow
            flow_es         = estimated_scaled_flows[0].squeeze(0).cpu().permute(1, 2, 0).numpy()
            
            # Get the groundturth and validty mask
            flow_mask_gt    = padding(item['optical_flow'])
            flow_gt         = flow_mask_gt[:,:2,:,:].squeeze(0).cpu().permute(1, 2, 0).numpy()
            mask            = flow_mask_gt[:,2,:,:].squeeze(0).cpu().numpy()
            

            # Get the groundturth and validty mask
            mask = np.expand_dims(mask, axis=-1)  # Shape becomes (H, W, 1)
            mask = np.repeat(mask, 2, axis=-1)  # Shape becomes (H, W, 2)

            masked_flow_es = flow_es * mask

            flow_es_vis = visualization_utils.flow_to_image(flow_es)
            masked_flow_es_vis = visualization_utils.flow_to_image(masked_flow_es)
            flow_gt_vis = visualization_utils.flow_to_image(flow_gt)
            
            txtManager.put_txt(flow_gt_vis,"GT Optical Flow")
            txtManager.put_txt(flow_es_vis,"Estimated Optical Flow")
            txtManager.put_txt(masked_flow_es_vis,"Masked Estimated Optical Flow")
                    
            out_img = cv2.hconcat([flow_gt_vis, flow_es_vis, masked_flow_es_vis])
            cv2.imshow("out_img", out_img)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            
            if idx > 100:
                break