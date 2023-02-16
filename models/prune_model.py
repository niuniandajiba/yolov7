import os
import torch
import torch.nn as nn
# import torch.nn.utils as utils
# import numpy as np
# from ..utils.torch_utils import prune, sparsity
from models.experimental import attempt_load
from pytorch2caffe import pytorch2caffe

def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a

def prune_model_l1unstructured(model, prune_rate=0.3):
    # Prune (currently unpruned) units in a tensor by zeroing out the ones with the lowest L1-norm
    # model: model to be pruned
    # prune_rate: prune rate
    # return: pruned model
    # Note: this function is only for unstructured pruning
    import torch.nn.utils.prune as prune
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=prune_rate)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))

def global_prune_model_l1unstructured(model, prune_rate=0.3):
    # Prune (currently unpruned) units in a tensor by zeroing out the ones with the lowest L1-norm
    # model: model to be pruned
    # prune_rate: prune rate
    # return: pruned model
    # Note: this function is only for unstructured pruning
    import torch.nn.utils.prune as prune
    prune.global_unstructured()
    prune.l1_unstructured(model, name='weight', amount=prune_rate)  # prune
    prune.remove(model, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))

if __name__ == '__main__':
    device = 'cpu'
    net_name = "yolov7_n2_tda2_ps_32st_20230104"
    weight_path = '/rd22857/repo/yolov7/runs/train/' + net_name + '/weights/last.pt'
    model = attempt_load(weight_path, map_location=device)
    model.model[-1].tda4 = True
    # prune_model_l1unstructured(model, prune_rate=0.2)
    dummy_input = torch.randn(1, 3, 640, 640, device=device)

    out_dir = '/rd22857/repo/yolov7/out_models'
    pytorch2caffe.trans_net(model, dummy_input, net_name)
    pytorch2caffe.save_prototxt(os.path.join(out_dir, net_name + ".prototxt"))
    pytorch2caffe.save_caffemodel(os.path.join(out_dir, net_name + ".caffemodel"))
    with open(os.path.join(out_dir, net_name + ".prototxt"), 'r') as f:
        prototxt = f.read()
        # prototxt = open(os.path.join(out_dir, net_name + ".prototxt"), 'r').read()
        prototxt = prototxt.replace('    round_mode: FLOOR\n', '')
        prototxt = prototxt.replace('input_dim: 1\ninput_dim: 3\ninput_dim: 640\ninput_dim: 640\n',
                    'input_shape {\n  dim: 1\n  dim: 3\n  dim: 640\n  dim: 640\n}\n')
    with open(os.path.join(out_dir, net_name + ".prototxt"), 'w') as f:
        f.write(prototxt)