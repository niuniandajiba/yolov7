docker run \
    --gpus all \
    --shm-size 32g \
    --ipc=host \
    --name env_rd22857_nv-pytorch \
    -it \
	-v /opt_disk1/share/Dataset:/dataset \
	-v /opt_disk2/rd22857:/rd22857 \
	nvcr.io/nvidia/pytorch:22.03-py3 \
	/bin/bash
	
# 	nvcr.io/nvidia/pytorch:${1}-py3 \