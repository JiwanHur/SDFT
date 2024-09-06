#!/bin/bash

gpu=$1
num_iter=$2
log_dirs=("./logs/metface_distill_8/" "./logs/metface_distill_9/" "./logs/metface_distill_10/")
for i in 0 1 2
do
    model_path="${log_dirs[i]}model${num_iter}.pt"
    echo $model_path
    ls_out= ls ${log_dirs[i]} | grep "model${num_iter}.pt"
    echo "ls out is $ls_out"
done

echo ${model_paths[0]}

# sample_dir=$3
# batch_size=32

# export PYTHONPATH=$PYTHONPATH:$(pwd)

# CUDA_VISIBLE_DEVICES=$gpu python scripts/image_sample.py \
#                         --attention_resolutions 16 \
#                         --class_cond False \
#                         --diffusion_steps 1000 \
#                         --dropout 0.0 \
#                         --image_size 256 \
#                         --learn_sigma True \
#                         --noise_schedule linear \
#                         --num_channels 128 \
#                         --num_res_blocks 1 \
#                         --num_head_channels 64 \
#                         --resblock_updown True \
#                         --use_fp16 True \
#                         --use_scale_shift_norm True \
#                         --timestep_respacing ddim40 \
#                         --model_path $model_path \
#                         --sample_dir $sample_dir \
#                         --batch_size $batch_size \
