#!/bin/bash

docker run -it --gpus=all --shm-size 64G --name hur \
-v ~:/workspace \
-v /mnt/disk1:/mnt \
-p 8866:8888 \
jwhur17/diff_pytorch \
/bin/bash