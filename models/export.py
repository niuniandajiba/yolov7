import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from pathlib import Path
from models.yolo import IDetect

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from utils.torch_utils import select_device
from utils.proto.pytorch2proto import prepare_model_for_layer_outputs, retrieve_onnx_names

from utils.proto import mmdet_meta_arch_pb2
from google.protobuf import text_format

def export_prototxt(model, img, file):
    # Prototxt export for a given ONNX model
    # prefix = colorstr('Prototxt:')
    onnx_model_name = str(file.with_suffix('.onnx'))

    for module in model.modules():
        if isinstance(module, IDetect):
            anchor_grid = torch.squeeze(module.anchor_grid)
            break
    num_heads = anchor_grid.shape[0]
    matched_names = retrieve_onnx_names(img, model, onnx_model_name)
    prototxt_name = onnx_model_name.replace('onnx', 'prototxt')

    background_label_id = -1
    num_classes = model.nc
    assert len(matched_names) == num_heads; "There must be a matched name for each head"
    proto_names = [f'{matched_names[i]}' for i in range(num_heads)]
    yolo_params = []
    for head_id in range(num_heads):
        yolo_param = mmdet_meta_arch_pb2.TIDLYoloParams(input=proto_names[head_id],
                                                        anchor_width=anchor_grid[head_id,:,0],
                                                        anchor_height=anchor_grid[head_id,:,1])
        yolo_params.append(yolo_param)

    nms_param = mmdet_meta_arch_pb2.TIDLNmsParam(nms_threshold=0.65, top_k=30000)
    detection_output_param = mmdet_meta_arch_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=mmdet_meta_arch_pb2.CODE_TYPE_YOLO_V5, keep_top_k=300,
                                            confidence_threshold=0.005)

    yolov5 = mmdet_meta_arch_pb2.TidlYoloOd(name='yolo_v5', output=["detections"],
                                            in_width=img.shape[3], in_height=img.shape[2],
                                            yolo_param=yolo_params,
                                            detection_output_param=detection_output_param,
                                            )
    arch = mmdet_meta_arch_pb2.TIDLMetaArch(name='yolo_v5', tidl_yolo=[yolov5])

    with open(prototxt_name, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolor-csp-c.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--tda4', action='store_true', help='export TDA4 model')
    parser.add_argument('--opset', type=int, default=11, help='opset_version')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    file = Path(opt.weights)
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # if isinstance(m, nn.Upsample):
        #     m.mode = 'bilinear'
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    model.model[-1].tda4 = opt.tda4  # set Detect() layer grid export
    # model.model[-1].stride = torch.tensor([32.0]).to(device)
    y = model(img)  # dry run

    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img, strict=False)
        ts.save(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=opt.opset, input_names=['images'],
                        #   operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
                        #   output_names=['classes', 'boxes'] if y is None else [], #'output0', 'output1', 'output2', 'output3'
                        #   dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                        #                 'output': {0: 'batch', 2: 'y', 3: 'x'}} if opt.dynamic else None
                                        )

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        # node = onnx_model.graph.node
        # for i in range(len(node)):
        #     inpname = node[i].input
        #     oupname = node[i].output
        #     for j in range(len(inpname)):
        #         for k in range(len(inpname[j])):
        #             if '0'<=inpname[j]<='9':
        #                 break
        #         if k == len(inpname[j])-1:
        #             continue
        #         node[i].input[j] = inpname[j][k:]
        #     for j in range(len(oupname)):
        #         for k in range(len(oupname[j])):
        #             if '0'<=oupname[j]<='9':
        #                 break
        #         if k == len(oupname[j])-1:
        #             continue
        #         node[i].output[j] = oupname[j][k:]
        # onnx.checker.check_model(onnx_model)  # check onnx model
        # onnx.save(onnx_model, f[:-5]+'2.onnx')
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)
    # export_prototxt(model, img, file)

    # CoreML export
    # try:
    #     import coremltools as ct

    #     print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
    #     # convert model from torchscript and apply pixel scaling as per detect.py
    #     model = ct.convert(ts, inputs=[ct.ImageType(name='image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
    #     f = opt.weights.replace('.pt', '.mlmodel')  # filename
    #     model.save(f)
    #     print('CoreML export success, saved as %s' % f)
    # except Exception as e:
    #     print('CoreML export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
