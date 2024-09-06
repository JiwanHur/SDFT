#!/bin/bash

gpu="0,1,2,3,4,5,6,7"
sh_file_name="train_ddp_distill.sh"
log_dir=${1}
distill_lambda=$2
distill_agnostic_lambda=$3
distill_feats=$4

export PYTHONPATH=$PYTHONPATH:$(pwd)

# CUDA_VISIBLE_DEVICES=$gpu python scripts/image_train.py \
mpiexec -n 8 python scripts/image_distill_train.py \
            --data_dir /mnt/disk1/metface/256/ \
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
            --batch_size 2 \
            --rescale_learned_sigmas True \
            --log_dir $log_dir\
            --sh_file_name $sh_file_name \
            --resume_checkpoint models/ffhq_p2.pt \
            --distill_lambda=$distill_lambda \
            --distill_p2_gamma=3 \
            --distill_agnostic=True \
            --distill_agnostic_gamma=100 \
            --distill_agnostic_lambda=$distill_agnostic_lambda \
            --distill_feats=$distill_feats \
            --distill_feats_lambda=0.05 \
            --last_iter=100k

        