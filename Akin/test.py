import os

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess, vis

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
    ):
        self.cls_names = COCO_CLASSES
        self.device = "mps"
        self.fp16 = False

        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size

        self.preproc = ValTransform(legacy=False)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16
        elif self.device == "mps":
            img = img.to(torch.device("mps"))

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

def convert_format(output, img_info):
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    if output is None:
        return img
    output = output.cpu()

    bboxes = output[:, 0:4]

    # preprocessing: resize
    bboxes /= ratio

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    new_boxes = []
    new_scores = []
    new_cls = []
    for i in range(len(cls)):
        if cls[i] == 0: # class 0 is person
            bbox = bboxes[i]
            x, y, w, h = bbox
            w = w - x
            h = h - y
            new_boxes.append(torch.tensor([x, y, w, h]))

            new_scores.append(scores[i])
            new_cls.append(cls[i])

    return new_boxes, new_scores, new_cls

def main(exp_file, ckpt_file, input_file, output_file):
    exp = get_exp(exp_file)

    exp.test_conf = 0.35
    exp.nmsthre = 0.45
    exp.test_size = (640, 640)

    model = exp.get_model()

    model.to(torch.device("mps"))
    model.eval()

    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])

    predictor = Predictor(model, exp)

    image_name = os.path.join(input_file)

    outputs, img_info = predictor.inference(image_name)

    result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
    cv2.imwrite(output_file, result_image)

    print(convert_format(outputs[0], img_info))


if __name__ == "__main__":
    exp_file = os.path.join("YOLOX", "exps", "default", "yolox_s.py")
    ckpt_file = os.path.join("yolox_s.pth")
    # input_file = os.path.join("YOLOX", "assets", "dog.jpg")
    input_file = os.path.join("000001.jpg")
    output_file = os.path.join("output.jpg")

    main(exp_file, ckpt_file, input_file, output_file)
