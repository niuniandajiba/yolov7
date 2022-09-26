model_name="yolov7_tiny_tda4_ps_32st_v2"
model_dir="./runs/train"
weight_dir="weights"
image_size_w="640"
image_size_h='640'
ckpt_name='last'
#H,W
python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w} --tda4
cd ${model_dir}/${model_name}/${weight_dir}
onnxsim ${ckpt_name}.onnx ${model_name}_sim.onnx
cd -
python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w} --grid
cd -
onnxsim ${ckpt_name}.onnx ${model_name}_grid_sim.onnx
cd -