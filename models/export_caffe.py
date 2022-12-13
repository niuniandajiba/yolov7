import torch
import torch.nn as nn
import os
import numpy as np
from pytorch2caffe import pytorch2caffe, caffe_net

from models.experimental import attempt_load
from models.common import inScale

if __name__ == '__main__':
    device = 'cpu'
    weight_path = './runs/train/yolov7_n_tda4_ps_32st_20221009/weights/last.pt'
    model = attempt_load(weight_path, map_location=device)
    model.model[-1].tda4 = True
    # model.model = nn.Sequential(inScale(), model.model)

    dummy_input = torch.randn(1, 3, 640, 640, device=device)

    out_dir = './out_models'
    pytorch2caffe.trans_net(model, dummy_input, "yolov7_n_tda4_ps_32st_20221009")
    
    batch_norm_layer = {
            "name": "batch_norm0", 
            "type": "BatchNorm", 
            "bottom": pytorch2caffe.log.cnet.net.input,
            "top": "batch_norm_blob0",
            "batch_norm_param" : {
                "use_global_stats": "true",
                "eps": 0.001
            }
        }

    layer1 = caffe_net.Layer_param(name=batch_norm_layer["name"],
                                    type=batch_norm_layer["type"],
                                    bottom=pytorch2caffe.log.cnet.net.input,
                                    top=[batch_norm_layer["top"]])
    layer1.batch_norm_param(use_global_stats=1, eps=0.001)
    running_mean = np.array([0.0]*3, np.float32)
    running_var = np.array([255.0*255.0]*3, np.float32)
    layer1.add_data(running_mean, running_var, np.array([1.0]))
    pytorch2caffe.log.cnet.add_layer(layer1)
    pytorch2caffe.log.cnet.net.layer[0].bottom[0] = "batch_norm_blob0"
    pytorch2caffe.save_prototxt(os.path.join(out_dir, "yolov7_n_tda4_ps_32st_20221009_bn.prototxt"))
    pytorch2caffe.save_caffemodel(os.path.join(out_dir, "yolov7_n_tda4_ps_32st_20221009_bn.caffemodel"))
