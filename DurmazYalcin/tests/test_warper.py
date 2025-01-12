"""
In order the to implement the photometric consistency loss, we need to have warper. 

Warper:
    - Inputs: An image frame and dense optical flow vectors.
    - Output: Warped version of image 1 through optical flow vectors. 
    
This python code is written to check if Warper really works. 
"""

from utils.utils import itemList
from utils.losses import warp_frame
import torch
import cv2
import numpy as np
import torch.nn.functional as F

def load_flow(path):
    with open(path, 'rb') as f:
        magic = float(np.fromfile(f, np.float32, count = 1)[0])
        if magic == 202021.25:
            w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            return data
        return None

flowList = itemList("WarperTestData/flow_vectors")
imagList = itemList("WarperTestData/image_frames")

for idx in range(imagList.itemCount()-2):
    imag_path = imagList.getItemPath(idx)
    imag_name = imagList.getItemName(idx)
    flow_path = flowList.getItemPathFromName(imag_name)
    
    next_image_path = imagList.getItemPath(idx+1)
    curr_image = cv2.imread(imag_path, cv2.IMREAD_GRAYSCALE)
    next_image = cv2.imread(next_image_path, cv2.IMREAD_GRAYSCALE)
    
    flow = load_flow(flow_path)
    flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)  # Shape: (B, 2, H, W)
    _, _, flow_height, flow_width = flow.size()  # Get flow dimensions
    
    curr_image = cv2.resize(curr_image, (flow_width, flow_height), interpolation=cv2.INTER_LINEAR)
    next_image = cv2.resize(next_image, (flow_width, flow_height), interpolation=cv2.INTER_LINEAR)

    curr_image_tensor = torch.from_numpy(curr_image).float().unsqueeze(0).unsqueeze(0)
    
    # Warp the current image
    warped_image_tensor = warp_frame(curr_image_tensor, flow)

    # Convert warped image tensor back to numpy for visualization
    warped_image = (warped_image_tensor.squeeze().cpu().numpy()).astype(np.uint8)  # Shape: (H, W)

    cv2.imshow("curr_image", curr_image)
    cv2.imshow("warped_image", warped_image)
    cv2.imshow("next_image", next_image)
    key = cv2.waitKey()
    if key == ord("q"):
        break