# 2022-0712-0851 6251MiB@tiny-32bts-640
# 2022-0712-1000 11765MiB@tiny-64bts-640 ds202207
model_name="yolov7_n_tda2_ps_16st_20221212"
weights="./pretrained_models/yolov7-tiny.pt"
python train_ps.py --worker 8 --device 0 --batch-size 64 --data data/ps_lh.yaml \
--cfg cfg/training/yolov7-n-tda2-ps-16st.yaml --weights ${weights} \
--name ${model_name} --hyp data/hyp.scratch.ps-tiny.yaml --epochs 300 --noautoanchor --strides16

model_dir="./runs/train"
weight_dir="weights"
image_size_w="384"
image_size_h='384'
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
python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w} --tda4
cd -
onnxsim ${ckpt_name}.onnx ${model_name}_tda4_sim.onnx
cd -
