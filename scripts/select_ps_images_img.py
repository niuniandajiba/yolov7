import os
import cv2
import math
import json
import base64
from tqdm import tqdm
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
    # run inference 
    model_path = '../onnx/yolov7_n2_tda2_ps_32st_20230113_grid_sim.onnx'
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    # model config
    conf_thres = 0.6
    dist_thres = 40
    inputSize = (640, 640)
    inMean = (0, 0, 0)
    inScale = (0.00392157, 0.00392157, 0.00392157)
    part_num = 1
    image_num = 0
    image_crop = [0, 960, 160, 1120] # [0, 960, 160, 1120]
    dir_batch_size = 1500
    cls_list = [[0, 'slot', (0, 255, 0)],
                [1, 'engaged', (0, 0, 255)]]
    # save config
    save_full_image = False
    shape_type = 'slot'
    # save_all_image = True
    # root path
    # img_dir = '/opt_disk2/rd22857/share_dataset/parking_slot_lh/raw/images/20221110131516'
    # ori_img_dir = '/opt_disk2/rd22857/share_dataset/parking_slot_lh/raw/images/20221110'
    # lbl_img_dir = '/opt_disk2/rd22857/share_dataset/parking_slot_lh/raw/images_lblshow/20221110'
    # lbl_json_dir = '/opt_disk2/rd22857/share_dataset/parking_slot_lh/raw/jsons/20221110'
    img_dir =      '/opt_disk2/rd22857/dataset/parkingslot_lh/process/raw/20230114'
    ori_img_dir =  '/opt_disk2/rd22857/dataset/parkingslot_lh/process/20230114'
    lbl_img_dir =  '/opt_disk2/rd22857/dataset/parkingslot_lh/process/20230114_lblshow'
    lbl_json_dir = '/opt_disk2/rd22857/dataset/parkingslot_lh/process/20230114'
    check_dir(ori_img_dir)
    check_dir(lbl_img_dir)
    check_dir(lbl_json_dir)
    # partdire path
    ori_img_partdir = os.path.join(ori_img_dir, 'part%02d' % part_num)
    lbl_img_partdir = os.path.join(lbl_img_dir, 'part%02d' % part_num)
    lbl_json_partdir = os.path.join(lbl_json_dir, 'part%02d' % part_num)
    check_dir(ori_img_partdir)
    check_dir(lbl_img_partdir)
    check_dir(lbl_json_partdir)

    for fname in tqdm(sorted(os.listdir(img_dir))):
        if fname.endswith('jpg'):
            image_num += 1
            img_path = os.path.join(img_dir, fname)
            # fname = '20221212_' + '%04d'%int(fname[6:-4]) + '.jpg'

            # check partnum < batchsize
            if image_num > dir_batch_size:
                part_num += 1
                ori_img_partdir = os.path.join(ori_img_dir, 'part%02d' % part_num)
                lbl_img_partdir = os.path.join(lbl_img_dir, 'part%02d' % part_num)
                lbl_json_partdir = os.path.join(lbl_json_dir, 'part%02d' % part_num)
                check_dir(ori_img_partdir)
                check_dir(lbl_img_partdir)
                check_dir(lbl_json_partdir)
                image_num = 1

            # copy to partdir
            ori_img_path = os.path.join(ori_img_partdir, fname)           

            # read image
            img = cv2.imread(img_path)
            img_ori = img.copy()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb[image_crop[0]:image_crop[1], image_crop[2]:image_crop[3]]
            
            # save img
            if save_full_image:
                if not os.path.exists(ori_img_path):
                    os.system('cp %s %s' % (img_path, ori_img_path))
            else:
                img_ori = img_ori[image_crop[0]:image_crop[1], image_crop[2]:image_crop[3]]
                img = img[image_crop[0]:image_crop[1], image_crop[2]:image_crop[3]]
                cv2.imwrite(ori_img_path, img_ori)

            img_rgb = cv2.resize(img_rgb, inputSize)
            img_tensor = preprocess(img_rgb, inMean, inScale)
            res0 = sess.run([output_name], {input_name: img_tensor})[0][0]
            res0 = res0[res0[:, 7] > conf_thres]

            # save show_label image
            lbl_img_path = os.path.join(lbl_img_partdir, fname)
            shapes = []
            if len(res0) == 0:
                # cp to lbl_img_partdir
                if save_full_image:
                    os.system('cp %s %s' % (img_path, lbl_img_path))
                else:
                    cv2.imwrite(lbl_img_path, img)
            else:
                slot = nms(dist_thres, res0)
                mask = np.zeros(img.shape, dtype=np.uint8)
                for idx in range(len(slot)):
                    delta_x = slot[idx][6] * slot[idx][2] / 2
                    delta_y = slot[idx][6] * slot[idx][3] / 2
                    if save_full_image:
                        p1 = ((slot[idx][0] + delta_x)*(image_crop[3]-image_crop[2])/inputSize[1]+image_crop[2], 
                                (slot[idx][1] + delta_y)*(image_crop[1]-image_crop[0])/inputSize[0]+image_crop[0])
                        p4 = ((slot[idx][0] - delta_x)*(image_crop[3]-image_crop[2])/inputSize[1]+image_crop[2],
                                (slot[idx][1] - delta_y)*(image_crop[1]-image_crop[0])/inputSize[0]+image_crop[0])
                    else:
                        p1 = ((slot[idx][0] + delta_x)*(image_crop[3]-image_crop[2])/inputSize[1],
                                (slot[idx][1] + delta_y)*(image_crop[1]-image_crop[0])/inputSize[0])
                        p4 = ((slot[idx][0] - delta_x)*(image_crop[3]-image_crop[2])/inputSize[1],
                                (slot[idx][1] - delta_y)*(image_crop[1]-image_crop[0])/inputSize[0])
                    if slot[idx][0]<0.5*inputSize[1]:
                        tmp = p1
                        p1 = p4
                        p4 = tmp
                    delta_x = slot[idx][4]*80
                    delta_y = slot[idx][5]*80
                    p2 = (p1[0]+delta_x, p1[1]+delta_y)
                    p3 = (p4[0]+delta_x, p4[1]+delta_y)
                    if shape_type == 'polygon':
                        points = np.array([p1, p2, p3, p4], dtype=np.float32)
                    elif shape_type == 'slot':
                        points = np.array([p4, p1, p2, p3], dtype=np.float32)
                    else:
                        raise ValueError('shape_type must be polygon or slot')
                    points_int = points.astype(np.int32)
                    # print(points_int)
                    cls_id = np.argmax(slot[idx][8:])
                    shapes.append({'label': cls_list[cls_id][1],
                                    'points': points.tolist(),
                                    'group_id': int(cls_id+1),
                                    'shape_type': shape_type,
                                    'flags': {}})
                    cls_color = cls_list[cls_id][2]
                    cv2.fillPoly(mask, [points_int], cls_color)
                    cv2.line(img, points_int[0], points_int[3], (0, 255, 0), 2)
                    cv2.line(img, points_int[0], points_int[1], (0, 0, 255), 2)
                    cv2.line(img, points_int[2], points_int[3], (0, 0, 255), 2)
                    cv2.circle(img, points_int[0], 6, (255, 0, 0), -1)
                cv2.addWeighted(img, 1, mask, 0.2, 0, img)
                cv2.imwrite(lbl_img_path, img)

            # save json           
            lbl_json_path = os.path.join(lbl_json_partdir, fname.replace('jpg', 'json'))
            json_dict = {
                    'version': '1.2.1',
                    'flags': {},
                    'shapes': shapes,
                    'imagePath': fname,
                    'imageData': base64.b64encode(cv2.imencode('.jpg', img_ori)[1]).decode('utf-8'),
                    'imageHeight': img_ori.shape[0],
                    'imageWidth': img_ori.shape[1]
                }
            json.dump(json_dict, open(lbl_json_path, 'w'), indent=2)
