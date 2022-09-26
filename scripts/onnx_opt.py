import onnx
import numpy as np

def main():
    model_path = './onnx/yolov7_tiny_tda4_ps_32st9_sim.onnx'
    tmp_model_path = './onnx/tmp.onnx'
    out_model_path = './onnx/yolov7_ps_640_640_pic953_32st_sig___.onnx'
    model = onnx.load(model_path)
    # onnx.save(model, tmp_model_path)
    # print('Done')
    # model = onnx.load(model_path)
    # graph = model.graph
    # input_names = ['images']
    # output_names = ['onnx::Reshape_256']
    # onnx.utils.extract_model(model_path, tmp_model_path, input_names, output_names)
    # print('stage 1 Done')
    # model = onnx.load(tmp_model_path)
    graph = model.graph
    # sig_node = onnx.helper.make_node('Sigmoid', ['onnx::Reshape_256'], ['999'])
    out_scale = np.array([64., 64., 2., 2., 2., 2., 192., 1., 1., 1., 
                  64., 64., 2., 2., 2., 2., 448., 1., 1., 1. ]).astype(np.float32)
    out_bias = np.array([0, 0, -1, -1, -1, -1, 0, 0, 0, 0,
                     0, 0, -1, -1, -1, -1, 0, 0, 0, 0 ]).astype(np.float32)
    out_mean = np.zeros([20]).astype(np.float32)
    out_var = np.ones([20]).astype(np.float32)
    # y = _batchnorm_test_mode(x, s, bias, mean, var).astype(np.float32)
    scale = onnx.helper.make_tensor('scale', onnx.TensorProto.FLOAT, [20], out_scale)
    bias = onnx.helper.make_tensor('bias', onnx.TensorProto.FLOAT, [20], out_bias)
    mean = onnx.helper.make_tensor('mean', onnx.TensorProto.FLOAT, [20], out_mean)
    var = onnx.helper.make_tensor('var', onnx.TensorProto.FLOAT, [20], out_var)
    

    bn_node = onnx.helper.make_node(
        "BatchNormalization",
        inputs=["257", "scale", "bias", "mean", "var"],
        outputs=["258"],
        name="BatchNormalization",
        epsilon=1e-5,
    )
    
    graph.initializer.extend([scale, bias, mean, var])
    graph.node.append(bn_node)
    graph.output.remove(graph.output[0])
    graph.output.insert(0, onnx.helper.make_tensor_value_info('258', onnx.TensorProto.FLOAT, [1, 20, 20, 20]))
    onnx.checker.check_model(model)
    onnx.save(model, out_model_path)
    print('stage 2 Done')

if __name__ == '__main__':
    main()