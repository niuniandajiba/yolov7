# 2022-0712-0851 6251MiB@tiny-32bts-640
# 2022-0712-1000 11765MiB@tiny-64bts-640 ds202207
python train.py --worker 8 --device 0 --batch-size 16 --data data/jiliod.yaml \
--img 1280 640 --cfg cfg/training/yolov7-tda4.yaml --weights '' \
--name yolov7_tda4 --hyp data/hyp.scratch.p5.yaml --epochs 200