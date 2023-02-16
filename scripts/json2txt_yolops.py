import os
import shutil
import json
import numpy as np
import cv2
from tqdm import tqdm
import math

dict_img = {'slot' : '0',
            'engaged' : '1'}
img_dir = '/opt_disk2/rd22857/dataset/parkingslot_lh/process/20221212/image'
#'/opt_disk1/share/Dataset/parking_slot_lh/金康/images'
json_dir = '/opt_disk2/rd22857/dataset/parkingslot_lh/process/20221212/json'
#'/opt_disk1/share/Dataset/parking_slot_lh/金康/jsons'
image_outdir = './20221212/images'
label_outdir = './20221212/labels'

theta_thres = 0.15
prefix = '20221212'

for fname in tqdm(os.listdir(json_dir)):
    if not fname.endswith('.json'):
        continue
    json_fpath = os.path.join(json_dir, fname)
    image_fpath = os.path.join(img_dir, fname[:-5]+'.jpg')
    image_outp = os.path.join(image_outdir, prefix+fname[:-5]+'.jpg')
    label_outp = os.path.join(label_outdir, prefix+fname[:-5]+'.txt')
    
    img = cv2.imread(image_fpath)
    left_border = 180
    right_border = 1100
    top_border = 40
    img = img[top_border:, left_border:right_border]
    left_add = 0
    right_add = 0
    top_add = 0
    bottom_add = 0
    img = cv2.copyMakeBorder(img, top_add, bottom_add, left_add, right_add, cv2.BORDER_CONSTANT, value=[100,100,100])
    cv2.imwrite(image_outp, img)
    
    # print(fname)
    with open(json_fpath, 'r') as f:
        data = json.load(f)
    image_width = 920
    image_height = 920

    label_f = open(label_outp, 'w')
    obj_list = data['shapes']
    label_f.write(str(image_width) + '\t' + str(image_height) + '\t' + str(len(obj_list)) + '\t\n')
    for i, obj in enumerate(obj_list):
        label_id = dict_img[obj['label']]
        points = (np.array(obj['points'])+0.5).astype(int)  # 0-based
        points[:, 0] -= left_border
        points[:, 1] -= top_border
        points[:, 0] += left_add
        points[:, 1] += top_add
        p1x, p1y = points[0]
        p2x, p2y = points[1]
        p3x, p3y = points[2]
        p4x, p4y = points[3]        
        if (p1x<0)|(p4x>image_width)|(p1x<0)|(p4x>image_width):
            print('Warning: out of range')
            print(fname)
            continue
        xmin = min(p1x, p2x, p3x, p4x)
        ymin = min(p1y, p2y, p3y ,p4y)
        xmax = max(p1x, p2x, p3x, p4x)
        ymax = max(p1y, p2y, p3y ,p4y)
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
        label_f.write(str(label_id) + '\t')
        label_f.write(str(xmin) + '\t' + str(ymin) + '\t')
        label_f.write(str(xmax) + '\t' + str(ymax) + '\t')
        label_f.write(str(p1x) + '\t' + str(p1y) + '\t')
        label_f.write(str(p2x) + '\t' + str(p2y) + '\t')
        label_f.write(str(p3x) + '\t' + str(p3y) + '\t')
        label_f.write(str(p4x) + '\t' + str(p4y) + '\t')
        label_f.write('\n')
    label_f.close()

# train_txt = open('../train.txt', 'w')
# for fname in os.listdir(label_outdir):
#     train_txt.write(fname[:-4] + '\n')
# train_txt.close()

