#!/bin/bash

gpu="0"
sh_file_name="train.sh"
export PYTHONPATH=$PYTHONPATH:$(pwd)
exp="aahq_limited1400_scratch"
CUDA_VISIBLE_DEVICES=$gpu python scripts/image_train.py \
                                --data_dir /mnt/raid/aahq_expressiveness_man_noglasses \
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
                                --batch_size 12 \
                                --rescale_learned_sigmas True \
                                --log_dir logs/$exp \
                                --sh_file_name $sh_file_name \
                                --save_interval 10000 \
                                --gpu $gpu 
                                # --resume_checkpoint models/ffhq_p2.pt
# additional

source_dir='/mnt/raid/aahq_256/'
epochs="010000 020000 040000 060000 080000"

for epoch in $epochs
do
    sample_dir='/home/jiwan.hur/samples_aahq/'$exp'_'$epoch'/images'
    bash sampling.sh $gpu ./logs/$exp/model$epoch.pt $sample_dir
    python metric.py --gpu $gpu \
                    --source_dir $source_dir \
                    --sample_dir $sample_dir \
                    --use_dataparallel False
done