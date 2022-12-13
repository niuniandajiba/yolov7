# 2022-0712-0851 6251MiB@tiny-32bts-640
# 2022-0712-1000 11765MiB@tiny-64bts-640 ds202207
model_name="yolov7_n_tda4_ps_32st_qat"
pretrained_weights="./runs/train/yolov7_n_tda4_ps_32st_20221009/weights/epoch_299.pt"
data="data/ps_lh_host.yaml"
hyp="data/hyp.scratch.ps-tiny_QAT.yaml"
cfg="cfg/training/yolov7-n-tda4-ps-32st.yaml"
python train_ps_qat.py --worker 8 --device 0 --batch-size 16 \
--data ${data} --cfg ${cfg} --weights ${pretrained_weights} \
--name ${model_name} --hyp ${hyp} --epochs 10 --noautoanchor

# below is done in train_ps_qat.py
# -----------------------------------
# model_dir="./runs/train"
# weight_dir="weights"
# image_size_w="640"
# image_size_h='640'
# ckpt_name='last'
#H,W
# python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w}
# cd ${model_dir}/${model_name}/${weight_dir}
# onnxsim ${ckpt_name}.onnx ${model_name}_sim.onnx
# cd -
# python ./models/export.py --weights ${model_dir}/${model_name}/${weight_dir}/${ckpt_name}.pt --img-size ${image_size_h} ${image_size_w} --grid
# cd -
# onnxsim ${ckpt_name}.onnx ${model_name}_grid_sim.onnx
# cp ${model_name}_grid_sim.onnx ../../../../onnx/
# cd -
