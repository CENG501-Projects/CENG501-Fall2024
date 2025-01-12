import os
import torch.nn as nn
import torch.nn.functional as F
import torch

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
    

def padding(tensor):
    # Make sure that the height and width of the input tenser is divisable by 16
    B, C, H, W = tensor.shape

    # Calculate padding for height and width
    pad_h = (16 - H % 16) % 16  # Padding needed for height
    pad_w = (16 - W % 16) % 16  # Padding needed for width

    # Apply padding: (left, right, top, bottom)
    padded_tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return padded_tensor

####### Adapted From Spike FlowNet #######
def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

####### Adapted From Spike FlowNet #######
def conv_s(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
        )

####### Adapted From Spike FlowNet #######
def predict_flow(batchNorm, in_planes):
    if batchNorm:
        return nn.Sequential(
                nn.BatchNorm2d(32),
                nn.Conv2d(in_planes,2,kernel_size=1,stride=1,padding=0,bias=False),
            )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, 2, kernel_size=1, stride=1, padding=0, bias=False),
        )

####### Adapted From Spike FlowNet #######
def deconv(batchNorm, in_planes, out_planes):
    if batchNorm:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True))

####### Adapted From Spike FlowNet #######
def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]