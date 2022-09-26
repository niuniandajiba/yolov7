# 2022-0712-0851 6251MiB@tiny-32bts-640
# 2022-0712-1000 11765MiB@tiny-64bts-640 ds202207
model_name="yolov7_n_tda4_ps_32st"
python train_ps.py --worker 8 --device 0 --batch-size 32 --data data/ps_lh.yaml \
--cfg cfg/training/yolov7-n-tda4-ps-32st.yaml --weights '' \
--name ${model_name} --hyp data/hyp.scratch.ps-tiny.yaml --epochs 300 --noautoanchor

model_dir="./runs/train"
weight_dir="weights"
image_size_w="640"
image_size_h='640'
ckpt_name='last'
#H,W
python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w}
cd ${model_dir}/${model_name}/${weight_dir}
onnxsim ${ckpt_name}.onnx ${model_name}_sim.onnx
cd -
python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w} --grid
cd -
onnxsim ${ckpt_name}.onnx ${model_name}_grid_sim.onnx
cp ${model_name}_grid_sim.onnx ../../../../onnx/
cd -
