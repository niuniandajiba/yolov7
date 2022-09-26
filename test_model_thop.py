import torch
import thop
from models.yolo import Model

mod = Model(cfg='./cfg/training/yolov7-tiny-tda4-ps-32st-v2.yaml')
dummy_input = torch.randn(1, 3, 640, 640)
flops, params = thop.profile(mod, inputs=(dummy_input, ))
macs, params = thop.clever_format([flops, params], "%.3f")
print('macs:', macs)
print('params:', params)
torch.onnx.export(mod, dummy_input, './test_net.onnx', opset_version=11)
