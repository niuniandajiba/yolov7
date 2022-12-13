# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 17:17:32 2022

@author: 6000022857
"""

import os
import cv2
import onnxruntime
import numpy as np
import math

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def preprocess(img_rgb, inMean, inScale):
    img = np.array(img_rgb).astype(np.float32)
    img -= inMean
    img *= inScale
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def main(img_dir, fname, session, inMean, inScale, outPath):
    img_outputName = os.path.join(outPath, fname)
    img_path = os.path.join(img_dir, fname)
    # Preprocess
    img_bgr = cv2.imread(img_path)
    # print(img_bgr.shape)
    if img_bgr.shape[1]>1000:
        img_bgr = img_bgr[:, 160:1120, :]
    img_bgr = cv2.resize(img_bgr, inputSize)
    tensor = preprocess(img_bgr[:, :, ::-1], inMean, inScale)
    
    # Run onnx session
    res0 = session.run(None, {'images':tensor})[0][0]
    res0 = res0[res0[:, 7]>0.5]
    # print(len(res0))
    res = sorted(res0, key=lambda res:res[7], reverse=True)
    
    # Threshold->NMS
    conf_thres = 0.6
    dist_thres = 40
    if len(res) == 0:
        valid_num=0
    for valid_num in range(len(res)):
        if res[valid_num][7] < conf_thres:
            break
    valid = np.ones(valid_num, dtype=int)
    slot = []
    for i in range(valid_num):
        if not valid[i]:
            continue
        slot.append(res[i])
        for j in range(i+1, valid_num):
            if not valid[j]:
                continue
            if math.dist(res[i][0:2], res[j][0:2])<dist_thres:
                valid[j]=0
    
    # Draw
    cls_list = [[0, 'Slot', (0, 255, 0)],
             [1, 'Engaged'  , (0, 0, 255)] ]
    for i in range(len(slot)):
        deltax = slot[i][6]*slot[i][2]/2
        deltay = slot[i][6]*slot[i][3]/2
        p1 = (int(slot[i][0]+deltax), int(slot[i][1]+deltay))
        p2 = (int(slot[i][0]-deltax), int(slot[i][1]-deltay))
        
        # dist_p1_p2 = math.dist(p1, p2)
        # if dist_p1_p2 > 150:
        #     len2 = 96
        # else:
        #     len2 = 192
        len2 = 30
        deltax = len2*slot[i][4]
        deltay = len2*slot[i][5]
        p3 = (int(p1[0]+deltax), int(p1[1]+deltay))
        p4 = (int(p2[0]+deltax), int(p2[1]+deltay))
        
        cls_id = np.argmax(slot[i][8:])
        cls_text = cls_list[cls_id][1]
        cls_color = cls_list[cls_id][2]
        text = cls_text + '_%3.2f%%' % (slot[i][7]*100*slot[i][8+cls_id])
        cv2.line(img_bgr, p1, p2, cls_color, thickness=2)
        cv2.line(img_bgr, p1, p3, cls_color, thickness=2)
        cv2.line(img_bgr, p2, p4, cls_color, thickness=2)
        # cv2.line(img_bgr, p3, p4, cls_color, thickness=2)
        cv2.putText(img_bgr, text, p1, cv2.FONT_HERSHEY_SIMPLEX,\
            fontScale=0.5, color=cls_color, thickness=1,\
            lineType=cv2.LINE_AA)
    
    cv2.imwrite(img_outputName, img_bgr)

if __name__ == '__main__':
    img_path = '20220919005628_4555.jpg'
    inputSize = (384, 384)
    inMean = (0, 0, 0)
    inScale = (0.00392157, 0.00392157, 0.00392157)
    model_path = './onnx/yolov7_n_tda2_ps_16st_20221128_grid_sim.onnx'
    session = onnxruntime.InferenceSession(model_path)
    # img_dir = '/rd22857/dataset/parkingslot_lh/images_test/1'
    img_dir = '/dataset/parking_slot_lh/raw/images/20221110131516'

    if 0: # Test
        main('./', img_path, session, inMean, inScale, './output_single')

    if 1:# batch test
        outPath = './output_test_20221110131516'
        check_dir(outPath)
        for fname in os.listdir(img_dir):
            if not os.path.exists(outPath):
                os.makedirs(outPath)
            main(img_dir, fname, session, inMean, inScale, outPath)
            print(fname)
    
    if 0:# video test
        outPath = './output_video'
        if not os.path.exists(outPath):
            os.makedirs(outPath)