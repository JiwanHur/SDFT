#!/bin/bash

gpu="0"
sh_file_name="train_ddp.sh"

exp="aahq_limited_ver2_transfer"

export PYTHONPATH=$PYTHONPATH:$(pwd)

# CUDA_VISIBLE_DEVICES=$gpu python scripts/image_train.py \
mpiexec -n 2 python scripts/image_train.py \
            --data_dir /mnt/raid/aahq_limited \
            --attention_resolutions 16 \
            --class_cond False \
            --diffusion_steps 1000 \
            --dropout 0.1 \
            --image_size 256 \
            --learn_sigma True \
            --noise_schedule linear \
            --num_channels 128 \
            --num_head_channels 64 \
            --num_res_blocks 1 \
            --resblock_updown True \
            --use_fp16 True \
            --use_scale_shift_norm True \
            --lr 2e-5 \
            --batch_size 8 \
            --rescale_learned_sigmas True \
            --log_dir logs/$exp \
            --sh_file_name $sh_file_name \
            --gpu $gpu \
            --save_interval 10000 \
            --resume_checkpoint models/ffhq_p2.pt 

# additional

source_dir='/mnt/raid/aahq_256/'
epochs="010000 020000 040000 060000 080000"
gpu=${gpu:0:1}

for epoch in $epochs
do
    sample_dir='/home/jiwan.hur/samples/'$exp'_'$epoch'/images'
    bash sampling.sh $gpu ./logs/$exp/model$epoch.pt $sample_dir
    python metric.py --gpu $gpu \
                    --source_dir $source_dir \
                    --sample_dir $sample_dir \
                    --use_dataparallel False
done