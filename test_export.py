import torch
from models.yolo import Model

mod = Model('./cfg/training/yolov7-tiny-tda4-ps.yaml')
dummy_input = torch.randn((1,3,960, 1280))
torch.onnx.export(mod, dummy_input, 'test_net.onnx', opset_version=11)