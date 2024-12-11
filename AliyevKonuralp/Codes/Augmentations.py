import random
from PIL import ImageOps
from torchvision import datasets, models, transforms
from PIL import Image
from PIL import ImageFilter
import numpy as np
class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    def __init__(self, threshold):
        assert 0 < threshold < 1
        self.threshold = round(threshold * 256)

    def __call__(self, img):
        return ImageOps.solarize(img, self.threshold)

    def __repr__(self):
        attrs = f"(min_scale={self.threshold}"
        return self.__class__.__name__ + attrs

class ResizeWithAspectRatio:
    def __init__(self, shorter_side=600):
        self.shorter_side = shorter_side

    def __call__(self, img):
        w, h = img.size
        if w < h:  # Width is shorter
            new_w = self.shorter_side
            new_h = int(h * (new_w / w))
        else:      # Height is shorter
            new_h = self.shorter_side
            new_w = int(w * (new_h / h))

        
        # Resize the image
        img_resized = img.resize((new_w, new_h), Image.ANTIALIAS)
        return img_resized

def build_strong_augmentation(is_train):
    augmentation = []
    if is_train:
        augmentation.append(
            ResizeWithAspectRatio(shorter_side=600)
        )
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))
        augmentation.append(transforms.RandomApply([Solarize(threshold=0.5)],
                                                   p=0.2))
        augmentation.append(transforms.ToTensor())
    else:
        augmentation.append(
           ResizeWithAspectRatio(shorter_side=600)
        )
        augmentation.append(transforms.ToTensor())
    return transforms.Compose(augmentation)


def AdjustBoundingBoxes(boxes, scale_x, scale_y):
    boxes_resized = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_min = x_min * scale_x
        y_min = y_min * scale_y
        x_max = x_max * scale_x
        y_max = y_max * scale_y
        boxes_resized.append([x_min, y_min, x_max, y_max])
    return boxes_resized


def ResizeScaleInformations(img,shorter_side=600):

    w, h = img.size
    if w < h:  # Width is shorter
        new_w = shorter_side
        new_h = int(h * (new_w / w))
    else:      # Height is shorter
        new_h = shorter_side
        new_w = int(w * (new_h / h))

    # Calculate scaling factors
    scale_x = new_w / w
    scale_y = new_h / h

    return  scale_x, scale_y

def validate_and_filter_boxes(target):
    

    valid_boxes = []
    valid_labels = []
    # Filter the bounding boxes and labels
    for box, label in zip(target['boxes'], target['labels']):
        if (box[2] > box[0]) and (box[3] > box[1]):  # Width and height > 0
            valid_boxes.append(box)
            valid_labels.append(label)
        
    if valid_boxes:
        valid_targets = {
            "boxes": np.array(valid_boxes, dtype=np.float32),  # Ensure consistent dtype
            "labels": np.array(valid_labels, dtype=np.int64),  # Ensure consistent dtype
            }
    return valid_targets