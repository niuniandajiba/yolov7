model_name="yolov7_n_tda2_ps_32st_20221117"
model_dir="./runs/train"
weight_dir="weights"
image_size_w="512"
image_size_h='512'
ckpt_name='last'
#H,W
python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w}
cd ${model_dir}/${model_name}/${weight_dir}
onnxsim ${ckpt_name}.onnx ${model_name}_512_sim.onnx
cd -
python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w} --tda4
cd -
onnxsim ${ckpt_name}.onnx ${model_name}_512_grid_sim.onnx
cd -
python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w} --grid
cd -
onnxsim ${ckpt_name}.onnx ${model_name}_512_tda4_sim.onnx
cp ${model_name}_tda4_sim.onnx ../../../../onnx/
cd -