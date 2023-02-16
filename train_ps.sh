# 2022-0712-0851 6251MiB@tiny-32bts-640
# # 2022-0712-1000 11765MiB@tiny-64bts-640 ds202207
# n-0.5-32st-640 loss: 0.0037
# n-0.5-16st-384 loss: 0.006 
# n-0.375-32st-640 loss: 0.005
model_name="yolov7_n2_tda2_ps_32st_20230113_375"
weights="./pretrained_models/yolov7_n_ps_20221212.5.pt"
python train_ps.py --worker 8 --device 0 --batch-size 64 --data data/ps_lh.yaml \
--cfg cfg/training/yolov7-n-tda2-ps-32st.yaml --weights ${weights} \
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
python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w} --tda4
cd -
onnxsim ${ckpt_name}.onnx ${model_name}_tda4_sim.onnx
cd -

# 2022-1219-1130 d6-960 32bts:oom 16bts:1epoch->oom 12bts √
# loss 0.004 
# model_name="yolov7_d_ps_32st_20221219"
# weights="./pretrained_models/yolov7-d6.pt"
# python train_ps.py --worker 8 --device 0 --batch-size 12 --data data/ps_lh.yaml \
# --cfg cfg/training/yolov7-d6_ps.yaml --weights ${weights} \
# --name ${model_name} --hyp data/hyp.scratch.ps-d6.yaml --epochs 300 --noautoanchor \
# --in960

# model_dir="./runs/train"
# weight_dir="weights"
# image_size_w="960"
# image_size_h='960'
# ckpt_name='last'
# #H,W
# python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w}
# cd ${model_dir}/${model_name}/${weight_dir}
# onnxsim ${ckpt_name}.onnx ${model_name}_sim.onnx
# cd -
# python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w} --grid
# cd -
# onnxsim ${ckpt_name}.onnx ${model_name}_grid_sim.onnx
# cp ${model_name}_grid_sim.onnx ../../../../onnx/
# cd -
# python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w} --tda4
# cd -
# onnxsim ${ckpt_name}.onnx ${model_name}_tda4_sim.onnx
# cd -
