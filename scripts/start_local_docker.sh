docker run \
    --gpus all \
    --shm-size 32g \
    --ipc=host \
    --name env_rd22857_yolov7 \
    -it \
	-v /opt_disk1/share/Dataset/jili_OD_VOC:/jiliod \
	-v ~/repo/yolov7:/yolov7 \
	nvcr.io/nvidia/pytorch:${1}-py3 \
	/bin/bash