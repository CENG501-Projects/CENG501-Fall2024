import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from copy import deepcopy
import torch_dct as dct
import jpegio as jio

file_paths = ['../CIMD/CIMD-R/dir_copy.txt',
              '../CIMD/CIMD-R/dir_remove.txt',
              '../CIMD/CIMD-R/dir_splicing.txt',]

def compute_recompression_coefficients(Q0, q, k=7):
    Qk = deepcopy(Q0)
    Q_list = [Qk]
    
    for _ in range(k):
        Dk = Qk * q
        Bk = dct.idct_2d(Dk)
        Ik1 = torch.round(Bk).clip(0, 255)
        DCT_Ik1 = dct.dct_2d(Ik1)
        Qk1 = DCT_Ik1 / q
        Qk1 = torch.round(Qk1)
        
        Qk = Qk1
        Q_list.append(Qk)
        
    return Q_list

def extract_y_channel_dct_and_q_matrix(image_path, device="cuda"):
    jpeg_image = jio.read(image_path)
    Q0 = jpeg_image.coef_arrays[0]
    print("Q0in utils: ", Q0)
    print("type of Q0in utils: ", type(Q0))
    
    q_matrix = jpeg_image.quant_tables[0]
    print("q_matrix in utils: ", q_matrix)
    print("type of q_matrix in utils: ", type(q_matrix))
    
    h_blocks = Q0.shape[0] // 8
    w_blocks = Q0.shape[1] // 8
    q = np.tile(q_matrix, (h_blocks, w_blocks))
    
    Q0_tensor = torch.tensor(Q0, dtype=torch.float32).to(device)
    q_tensor = torch.tensor(q, dtype=torch.float32).to(device)
    
    return Q0_tensor, q_tensor


def convert_to_binary_volume(Q0_tensor, T=20):
    Q0_clipped = torch.clamp(Q0_tensor, min=-T, max=T)
    Q0_abs = torch.abs(Q0_clipped).long()
    
    H, W = Q0_abs.shape
    binary_volume = torch.zeros((T + 1, H, W), dtype=torch.uint8)
    
    binary_volume = torch.nn.functional.one_hot(Q0_abs.view(-1), num_classes=T + 1)
    binary_volume = binary_volume.view(H, W, T + 1).permute(2, 0, 1)
    
    return binary_volume.float()

def compute_residual_dct(Q_list):
    R = torch.zeros_like(Q_list[0])
    for i in range(1, len(Q_list)):
        R += (Q_list[i] - Q_list[i - 1]) / len(Q_list)
    return R


def reshape_dct_blocks(tensor):
    
    batch_size, H, W = tensor.shape
    assert H % 8 == 0 and W % 8 == 0, "Height and Width must be multiples of 8."
    
    tensor = tensor.unfold(1, 8, 8).unfold(2, 8, 8)
    
    tensor = tensor.contiguous().view(batch_size, -1, 8, 8)
    tensor = tensor.view(batch_size, -1, 64)
    
    
    num_blocks_h = H // 8
    num_blocks_w = W // 8
    tensor = tensor.view(batch_size, num_blocks_h, num_blocks_w, 64)
    
    return tensor 

image_to_tensor =   transforms.Compose([
                        transforms.ToTensor()
                    ])

def get_io_paths(file_paths = file_paths):
    input_paths = []
    output_paths = []
    class_labels = []
    
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                line_array = line.strip().split(' ')
                image_path = "../CIMD/CIMD-R/" + line_array[0]
                mask_path = "../CIMD/CIMD-R/" + line_array[1]
                class_label = line_array[2]
                
                input_paths.append(image_path)
                output_paths.append(mask_path)
                class_labels.append(class_label)
    
    return input_paths, output_paths, class_labels

image_to_tensor =   transforms.Compose([
                        transforms.Resize((1344, 2048)),
                        transforms.ToTensor()
                        ])

def show_rgb_tensor(tensor):
    np_img = tensor.permute(1, 2, 0).numpy()  #! to HWC
    np_img = (np_img * 255).astype(np.uint8)  #! to uint8

    pil_image = Image.fromarray(np_img)
    pil_image.show()
    
    
def show_gs_tensor(tensor):
    np_img = tensor.permute(1, 2, 0).detach().numpy()  #! to HWC format
    np_img = (np_img * 255).astype(np.uint8)  #! to uint8
    np_img = np_img.squeeze()
    pil_image = Image.fromarray(np_img, mode='L')
    pil_image.show()

class ImageDataset(Dataset):
    def __init__(self, input_paths, output_paths, class_labels):
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.class_labels = class_labels
        
    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, idx):
        x = image_to_tensor(Image.open(self.input_paths[idx]))
        h, w = x.size(1), x.size(2)
        class_label = self.class_labels[idx]
        if class_label == '0':
            y = torch.zeros(1, h, w)
        else:
            y = image_to_tensor(Image.open(self.output_paths[idx]))
        return x, y, int(class_label)

def get_accuracy(preds, class_labels):
    pred = deepcopy(preds.detach())
    pred = torch.max(pred, dim=2).values
    pred = torch.max(pred, dim=2).values
    
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = pred.squeeze()
    
    acc = sum(pred == class_labels) / len(class_labels)
    return acc.item()

def logits_to_preds(tensor):
    tensor_ = deepcopy(tensor.detach())
    tensor_[tensor_ >= 0.5] = 1
    tensor_[tensor_ < 0.5] = 0
    return tensor_

def get_pixelwise_accuracy(y_true, y_pred):
    #! flatten tensors
    y_true_ = logits_to_preds(y_true).view(-1)
    y_pred_ = logits_to_preds(y_pred).view(-1)
    correct = torch.sum(y_true_ == y_pred_)
    total = y_true_.shape[0]
    accuracy = correct / total
    
    return accuracy.item()


def calculate_pixel_f1(y_true, y_pred, eps=1e-8, threshold=0.5):
    prediction = logits_to_preds(y_pred).view(-1)
    target = logits_to_preds(y_true).view(-1)
    
    if prediction.min() >= 0 and prediction.max() <= 1:
        prediction = (prediction > threshold).float()
    if target.min() >= 0 and target.max() <= 1:
        target = (target > threshold).float()
        
    true_positives = torch.sum(prediction * target)
    false_positives = torch.sum(prediction * (1 - target))
    false_negatives = torch.sum((1 - prediction) * target)
    
    precision = true_positives / (true_positives + false_positives + eps)
    recall = true_positives / (true_positives + false_negatives + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    
    return f1.item()