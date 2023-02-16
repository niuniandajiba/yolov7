# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:24:51 2022

@author: 6000022857
"""
import os
import cv2
import math
import json
import base64
import datetime
from tqdm import tqdm
# from glob import glob
import numpy as np
import onnxruntime as ort

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def preprocess(img_rgb, inMean, inScale):
    img = np.array(img_rgb).astype(np.float32)
    img -= inMean
    img *= inScale
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def nms(dist_thres, res0):
    res = sorted(res0, key=lambda x: x[7], reverse=True)
    valid = np.ones(len(res), dtype=int)
    slot = []
    for idx in range(len(res)):
        if not valid[idx]:
            continue
        slot.append(res[idx])
        for j in range(idx+1, len(res)):
            if not valid[j]:
                continue
            if math.dist(res[idx][0:2], res[j][0:2]) < dist_thres:
                valid[j] = 0
    return slot

if __name__ == '__main__':
    # model config
    conf_thres = 0.6
    dist_thres = 40
    inputSize = (640, 640)
    inMean = (0, 0, 0)
    inScale = (0.00392157, 0.00392157, 0.00392157)
    part_num = 1
    image_num = -1
    
    # configs
    video_crop = [40, 960, 180, 1100]
    dir_batch_size = 1000
    cls_list = [[0, 'slot', (0, 255, 0)],
                [1, 'engaged', (0, 0, 255)]]

    # datetime
    now = datetime.datetime.now()
    # now_str = now.strftime('%Y%m%d')
    now_str = '20220919'
    now_str_t = '20220919_pre'

    # path
    root = '/opt_disk1/share/Dataset/parking_slot_lh/raw'
    video_dir = os.path.join('videos', now_str)
    ori_img_dir = os.path.join('images', now_str_t)
    lbl_img_dir = os.path.join('images_lblshow', now_str)
    lbl_json_dir = os.path.join('jsons', now_str)
    model_path = 'utils/yolov7_tiny_tda4_ps_32st_grid_sim_pic953.onnx'
    video_dir = os.path.join(root, video_dir)
    ori_img_dir = os.path.join(root, ori_img_dir)
    lbl_img_dir = os.path.join(root, lbl_img_dir)
    lbl_json_dir = os.path.join(root, lbl_json_dir)
    check_dir(ori_img_dir)
    check_dir(lbl_img_dir)
    check_dir(lbl_json_dir)
    ori_img_partdir = os.path.join(ori_img_dir, 'part%02d' % part_num)
    lbl_img_partdir = os.path.join(lbl_img_dir, 'part%02d' % part_num)
    lbl_json_partdir = os.path.join(lbl_json_dir, 'part%02d' % part_num)
    check_dir(ori_img_partdir)
    check_dir(lbl_img_partdir)
    check_dir(lbl_json_partdir)
    model_path = os.path.join(root, model_path)
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    for id, video_fname in enumerate(os.listdir(video_dir)):
        print(str(id)+'.'+video_fname)
        if video_fname.endswith('.hdr'):
            continue
        video_path = os.path.join(video_dir, video_fname)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = 5
        # print(frame_count)
        # h264 frame_count -192153584101141
        # for i in tqdm(range(frame_count)):
        i = 0
        while True:
            i+=1
            ret, frame0 = cap.read()
            if ret & ((i%frame_interval) == 0):
                # cv2.imwrite(output_path, frame)
                frame = frame0.copy()
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb = img_rgb[video_crop[0]:video_crop[1], video_crop[2]:video_crop[3], :]
                img_rgb = cv2.resize(img_rgb, inputSize)
                img_tensor = preprocess(img_rgb, inMean, inScale)
                res0 = ort_session.run(['344'], {input_name: img_tensor})[0][0]
                res0 = res0[res0[:, 7] > conf_thres]
                # image with slots
                if res0.shape[0] > 0:
                    # 1. save original image
                    # --------------------------------
                    image_num += 1
                    if image_num >= dir_batch_size:
                        part_num += 1
                        ori_img_partdir = os.path.join(ori_img_dir, 'part%02d' % part_num)
                        check_dir(ori_img_partdir)
                        lbl_img_partdir = os.path.join(lbl_img_dir, 'part%02d' % part_num)
                        check_dir(lbl_img_partdir)
                        lbl_json_partdir = os.path.join(lbl_json_dir, 'part%02d' % part_num)
                        check_dir(lbl_json_partdir)
                        image_num = 0
                    output_path = os.path.join(ori_img_partdir, now_str+video_fname[14:22].replace('-', '')+'_'+str(i)+'.jpg')
                    cv2.imwrite(output_path, frame)     

                    # 2. save show_label image
                    # --------------------------------
                    slot = nms(dist_thres, res0)
                    mask = np.zeros(frame.shape, dtype=np.uint8)
                    shapes = []
                    # print(slot)
                    for idx in range(len(slot)):
                        delta_x = slot[idx][6] * slot[idx][2] / 2
                        delta_y = slot[idx][6] * slot[idx][3] / 2
                        p1 = ((slot[idx][0] + delta_x)*(video_crop[3]-video_crop[2])/inputSize[1]+video_crop[2], 
                              (slot[idx][1] + delta_y)*(video_crop[1]-video_crop[0])/inputSize[0]+video_crop[0])
                        p4 = ((slot[idx][0] - delta_x)*(video_crop[3]-video_crop[2])/inputSize[1]+video_crop[2],
                              (slot[idx][1] - delta_y)*(video_crop[1]-video_crop[0])/inputSize[0]+video_crop[0])
                        if slot[idx][0]<0.5*inputSize[1]:
                            tmp = p1
                            p1 = p4
                            p4 = tmp
                        delta_x = slot[idx][4]*80
                        delta_y = slot[idx][5]*80
                        p2 = (p1[0]+delta_x, p1[1]+delta_y)
                        p3 = (p4[0]+delta_x, p4[1]+delta_y)
                        points = np.array([p1, p2, p3, p4], dtype=np.float32)
                        points_int = points.astype(np.int32)
                        # print(points_int)
                        cls_id = np.argmax(slot[idx][8:])
                        shapes.append({'label': cls_list[cls_id][1],
                                       'points': points.tolist(),
                                       'group_id': int(cls_id+1),
                                       'shape_type': 'polygon',
                                       'flags': {}})
                        cls_color = cls_list[cls_id][2]
                        cv2.fillPoly(mask, [points_int], cls_color)
                        cv2.line(frame, points_int[0], points_int[3], (0, 255, 0), 2)
                        cv2.line(frame, points_int[0], points_int[1], (0, 0, 255), 2)
                        cv2.line(frame, points_int[2], points_int[3], (0, 0, 255), 2)
                        cv2.circle(frame, points_int[0], 6, (255, 0, 0), -1)
                    cv2.addWeighted(frame, 1, mask, 0.2, 0, frame)
                    output_path = os.path.join(lbl_img_partdir, now_str+video_fname[14:22].replace('-', '')+'_'+str(i)+'.jpg')
                    cv2.imwrite(output_path, frame)

                    # 3. save json label
                    # --------------------------------
                    json_dict = {
                        'version': '4.5.13',
                        'flags': {},
                        'shapes': shapes,
                        'imagePath': now_str+video_fname[14:22].replace('-', '')+'_'+str(i)+'.jpg',
                        'imageData': base64.b64encode(cv2.imencode('.jpg', frame0)[1]).decode('utf-8'),
                        'imageHeight': frame.shape[0],
                        'imageWidth': frame.shape[1]
                    }
                    output_path = os.path.join(lbl_json_partdir, now_str+video_fname[14:22].replace('-', '')+'_'+str(i)+'.json')
                    json.dump(json_dict, open(output_path, 'w'), indent=2)
            
            elif not ret:
                break
        cap.release()