import os
import cv2
import json
import base64
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class ParkingSlotDetector():
    def __init__(self, model_path, model_config):
        self.model_path = model_path
        self.model_config = model_config
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.classes = ['slot', 'engaged', '___deleted___']
    
    def preprocess(self, img_rgb, inMean, inScale):
        img = np.array(img_rgb).astype(np.float32)
        img -= inMean
        img *= inScale
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def nms_parkingslot(self, dist_thres, res0):
        res = sorted(res0, key=lambda x: x[7], reverse=True)
        valid = np.ones(len(res), dtype=int)
        slots = []
        for idx in range(len(res)):
            if not valid[idx]:
                continue
            slots.append(res[idx])
            for j in range(idx+1, len(res)):
                if not valid[j]:
                    continue
                if np.linalg.norm(np.array(res[idx][0:2]) - np.array(res[j][0:2])) < dist_thres:
                    valid[j] = 0
        return slots

    def postprocess(self, slot, scale_x, scale_y):
        delta_x = slot[6] * slot[2] / 2
        delta_y = slot[6] * slot[3] / 2
        p1 = ((slot[0]+delta_x)*scale_x, (slot[1]+delta_y)*scale_y)
        p2 = ((slot[0]-delta_x)*scale_x, (slot[1]-delta_y)*scale_y)
        delta_x = slot[4] * 100
        delta_y = slot[5] * 100
        p3 = (p2[0]+delta_x, p2[1]+delta_y)
        p4 = (p1[0]+delta_x, p1[1]+delta_y)
        points = np.array([p1, p2, p3, p4], dtype=np.float32)
        cls_id = np.argmax(slot[8:])
        return points, cls_id

    def slot2labelps(self, points, cls_id):
        return {
            'label' : self.classes[cls_id],
            'points' : points.tolist(),
            'group_id' : int(cls_id + 1),
            'shape_type' : 'slot',
            'flags' : {}
        }

    def detect(self, img):
        scale_x = img.shape[1] / self.model_config['inputSize'][0]
        scale_y = img.shape[0] / self.model_config['inputSize'][1]
        img = cv2.resize(img, self.model_config['inputSize'])
        img = self.preprocess(img, self.model_config['inMean'], self.model_config['inScale'])
        res = self.session.run([self.output_name], {self.input_name: img})[0][0]
        res = res[res[:, 7] > self.model_config['conf_thres']]
        res = self.nms_parkingslot(self.model_config['dist_thres'], res)
        

