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

if __name__ == '__main__':
    device = 'cpu'
    weight_path = '/rd22857/repo/yolov7/runs/train/yolov7_n_tda2_ps_16st_20221123/weights/last.pt'
    model = attempt_load(weight_path, map_location=device)
    model.model[-1].tda4 = True
    prune_model_l1unstructured(model, prune_rate=0.3)
    dummy_input = torch.randn(1, 3, 384, 384, device=device)

    out_dir = '/rd22857/repo/yolov7/out_models'
    pytorch2caffe.trans_net(model, dummy_input, "yolov7_n_tda2_ps_16st_20221123_spas")
    pytorch2caffe.save_prototxt(os.path.join(out_dir, "yolov7_n_tda2_ps_16st_20221123_spas.prototxt"))
    pytorch2caffe.save_caffemodel(os.path.join(out_dir, "yolov7_n_tda2_ps_16st_20221123_spas.caffemodel"))
