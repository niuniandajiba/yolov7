# 2022-0712-0851 6251MiB@tiny-32bts-640
# 2022-0712-1000 11765MiB@tiny-64bts-640 ds202207
python train_ps.py --worker 8 --device 0 --batch-size 2 --data data/ps.yaml \
--img 640 640 --cfg cfg/training/yolov7-tiny-tda4-ps.yaml --weights '' \
--name yolov7_tiny_tda4_ps --hyp data/hyp.scratch.ps-tiny.yaml --epochs 200