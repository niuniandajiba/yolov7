import onnx

if __name__ == '__main__':
    # slice model
    model_path = '/rd22857/repo/yolov7/onnx/yolov7_n_tda2_ps_32st_20221110_sim.onnx'
    out_path = '/rd22857/repo/yolov7/onnx/yolov7_n_tda2_ps_32st_20221110_sim.onnx'

    model = onnx.load(model_path)
    input_name = [model.graph.input[0].name]
    output_name = ["onnx::Reshape_264"]
    onnx.utils.extract_model(model_path, out_path, input_name, output_name)