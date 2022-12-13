# 2022-0712-0851 6251MiB@tiny-32bts-640
# 2022-0712-1000 11765MiB@tiny-64bts-640 ds202207
# python train.py --worker 8 --device 1 --batch-size 64 --data data/jiliod.yaml \
# --img 640 640 --cfg cfg/training/yolov7-tiny-tda4.yaml --weights '' \
# --name yolov7_tiny_tda4 --hyp data/hyp.scratch.tiny.yaml --epochs 200

# 2022-1115 yolov7-nano-tda2
model_name="yolov7_n_tda2_od_20221115"
weights="./pretrained_models/yolov7-tiny.pt"
python train.py --worker 8 --device 0 --batch-size 64 --data data/jiliod.yaml \
--img 1280 640 --cfg cfg/training/yolov7-n-tda2-od.yaml --weights ${weights} \
--name ${model_name} --hyp data/hyp.scratch.tiny.yaml --epoch 300

model_dir="./runs/train"
weight_dir="weights"
image_size_w="768"
image_size_h='384'
ckpt_name='last'
#H,W
python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w} --grid
cd ${model_dir}/${model_name}/${weight_dir}
onnxsim ${ckpt_name}.onnx ${model_name}_grid_sim.onnx
# cp ${model_name}_grid_sim.onnx ../../../../onnx/
cd -
python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w} --tda4
cd ${model_dir}/${model_name}/${weight_dir}
onnxsim ${ckpt_name}.onnx ${model_name}_tda4_sim.onnx
cp ${model_name}_tda4_sim.onnx ../../../../onnx/
cd -