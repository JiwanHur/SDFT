#!/bin/bash

gpu=$1
model_path=$2
sample_dir=$3
batch_size=128

export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=$gpu python scripts/image_sample.py \
                        --attention_resolutions 16 \
                        --class_cond False \
                        --diffusion_steps 1000 \
                        --dropout 0.0 \
                        --image_size 256 \
                        --learn_sigma True \
                        --noise_schedule linear \
                        --num_channels 128 \
                        --num_res_blocks 1 \
                        --num_head_channels 64 \
                        --resblock_updown True \
                        --use_fp16 True \
                        --use_scale_shift_norm True \
                        --timestep_respacing ddim40 \
                        --model_path $model_path \
                        --sample_dir $sample_dir \
                        --batch_size $batch_size \
                        --fixed_seed True \
                        