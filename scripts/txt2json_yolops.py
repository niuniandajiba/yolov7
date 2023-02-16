import os
import cv2
import json
import base64
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    # config 
    # slot polygon
    shape_type = 'slot'
    label_format = 'bin'

    cls_list = [[0, 'slot'],
                [1, 'engaged']]
    txt_dir = '/opt_disk2/rd22857/dataset/parkingslot_lh/labels_new'
    json_dir = '/opt_disk2/rd22857/dataset/parkingslot_lh/jsons_new'
    image_dir = '/opt_disk2/rd22857/dataset/parkingslot_lh/images'

    for fname in tqdm(os.listdir(txt_dir)):
        if not fname.endswith(label_format):
            print('-'*50)
            print('skip %s' % fname)
            continue
        txt_path = os.path.join(txt_dir, fname)
        json_path = os.path.join(json_dir, fname[:-4] + '.json')
        image_path = os.path.join(image_dir, fname[:-4] + '.jpg')
        shapes = []
        if label_format == 'txt':
            f = open(txt_path, 'r')
            f_lbl = f.readlines()[1:]
            f.close()
        elif label_format == 'bin':
            f_lbl = np.fromfile(txt_path, dtype=np.float32).reshape(-1, 9)
        for i in range(len(f_lbl)):
            if label_format == 'txt':
                label = np.array(f_lbl[i].split('\t')[:-1]).astype(float)
                cls_id = int(label[0])
                p1x, p1y = label[5], label[6]
                p2x, p2y = label[7], label[8]
                p3x, p3y = label[9], label[10]
                p4x, p4y = label[11], label[12]
            elif label_format == 'bin':
                label = f_lbl[i].astype(float)
                cls_id = int(label[0])
                p1x, p1y = label[1], label[2]
                p2x, p2y = label[3], label[4]
                p3x, p3y = label[5], label[6]
                p4x, p4y = label[7], label[8]
            if shape_type == 'slot':
                points = [[p4x, p4y], [p1x, p1y], [p2x, p2y], [p3x, p3y]]
            elif shape_type == 'polygon':
                points = [[p1x, p1y], [p2x, p2y], [p3x, p3y], [p4x, p4y]]
            else:
                raise ValueError('shape_type must be slot or polygon')
            shapes.append({
                'label': cls_list[cls_id][1],
                'points': points,
                'group_id': cls_id+1,
                'shape_type': shape_type,
                'flags': {}
            })
            
        image = cv2.imread(image_path)
        json_dict = {
            'version': '5.1.1',
            'flags': {},
            'shapes': shapes,
            'imagePath': '../images/' + fname[:-4] + '.jpg',
            'imageData': base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8'),
            'imageHeight': image.shape[0],
            'imageWidth': image.shape[1]
        }
        json.dump(json_dict, open(json_path, 'w'), indent=4)
                
