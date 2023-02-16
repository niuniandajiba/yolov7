import os
import cv2
import numpy as np
import json
import math

dire = '/opt_disk1/share/Dataset/parking_slot_lh/raw/images/20220919_pre/part04-2'
json_dir = '/opt_disk1/share/Dataset/parking_slot_lh/raw/jsons/20220919/part04-2-label'
outdir = '/opt_disk1/share/Dataset/parking_slot_lh/raw/images_lblshow/20220919_lbl1/part04'
theta_thres = 0.15

if not os.path.exists(outdir):
    os.makedirs(outdir)

def show_slot(dire, json_dir, outdir, theta_thres=0.15):
    dict_img = {'slot' : '1',
                'engaged' : '2'}
    color_list = [(0, 255, 0), (0, 0, 255)]
    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith('.json'):
            continue
        img_path = os.path.join(dire, fname[:-5]+'.jpg')
        json_path = os.path.join(json_dir, fname)
        img = cv2.imread(img_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
        obj_list = data['shapes']
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if len(obj_list)==0:
            cv2.putText(img, 'No object', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
        for i, obj in enumerate(obj_list):
            label_id = dict_img[obj['label']]
            points = (np.array([obj['points']])+0.5).astype(int)
        # print(points.shape)
            p1x, p1y = points[0][0]
            p2x, p2y = points[0][1]
            p3x, p3y = points[0][2]
            p4x, p4y = points[0][3]

            leng_1_2 = math.dist((p1x, p1y), (p2x, p2y))
            cos_theta3 = (p2x-p1x)/leng_1_2
            sin_theta3 = (p2y-p1y)/leng_1_2
            leng_3_4 = math.dist((p3x, p3y), (p4x, p4y))
            cos_theta4 = (p3x-p4x)/leng_3_4
            sin_theta4 = (p3y-p4y)/leng_3_4
        # Check |theta3-theta4|<...
            if (abs(sin_theta3-sin_theta4) > theta_thres)|(abs(cos_theta3-cos_theta4) > theta_thres):
                print('Warning:Incorrect slot information!')
                print('Filename:' + str(fname))
                print('SlotID:' + str(i))
                print('Delta:' + str(abs(sin_theta3-sin_theta4)) + ' ' + str(abs(cos_theta3-cos_theta4)))

            cv2.fillPoly(mask, points, int(label_id))
            cv2.line(img, (p1x, p1y), (p2x, p2y), (0, 0, 255), 2)
            cv2.line(img, (p2x, p2y), (p3x, p3y), (0, 0, 255), 2)
            cv2.line(img, (p3x, p3y), (p4x, p4y), (0, 0, 255), 2)
            cv2.line(img, (p4x, p4y), (p1x, p1y), (0, 255, 0), 2)
            cv2.circle(img, (p1x, p1y), 6, (255, 0, 0), -1)

        mask_color = np.zeros(img.shape, dtype=np.uint8)
        mask_color[mask==1] = color_list[0]
        mask_color[mask==2] = color_list[1]
        cv2.addWeighted(img, 1, mask_color, 0.2, 0, img)
        cv2.imwrite(os.path.join(outdir, fname[:-5]+'.jpg'), img)
        # print(fname)

show_slot(dire, json_dir, outdir, theta_thres)