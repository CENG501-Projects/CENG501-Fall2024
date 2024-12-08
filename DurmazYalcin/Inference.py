import torch

from SNN import SpikeFlow
from DSECLoader import DSECDataset, seq_train, seq_val
import cv2
import numpy as np

import visualization_utils
import time
import sys


from visualization_utils import txtAdder
    
    
if __name__ == "__main__":
    model      = SpikeFlow().cuda()
    model.load_state_dict(torch.load('checkpoints/best.pth', weights_only=True))
    
    main_path   = "/home/hakito/datasets/event_data/dsec"
    data = DSECDataset(main_path, seq_val)
    
    txtManager = txtAdder()
    
    for idx,item in enumerate(data):
        input_representation = item['input_tensors']
        torch_tensors = [torch.from_numpy(input) for input in input_representation]

        torch_tensors = [t.unsqueeze(0) for t in torch_tensors]
        input = torch.cat(torch_tensors, dim=1).float().cuda()
        
        estimated_scaled_flows = model(input)
        
        flow_es = estimated_scaled_flows[0].squeeze(0).cpu().detach().numpy()
        flow_es = converted_array = np.transpose(flow_es, (1, 2, 0)) # (C,H,W) to (H,W,C)
        flow_gt = item['uv_gt']
        
        mask = item['uv_validity']
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
        cv2.imwrite("ESvsGT/"+str(idx).zfill(3)+".png",out_img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        
        if idx > 100:
            break