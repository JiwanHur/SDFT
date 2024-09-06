#!/bin/bash

gpu="0,1,2,3,4,5,6,7"
sh_file_name="train_ddp_distill.sh"

exp='metface_limited_distill'

export PYTHONPATH=$PYTHONPATH:$(pwd)

mpiexec -n 2 python scripts/image_distill_train.py \
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
            --batch_size 2 \
            --rescale_learned_sigmas True \
            --log_dir logs/$exp \
            --sh_file_name $sh_file_name \
            --dist_checkpoint models/ffhq_p2.pt \
            --distill_lambda=0.1 \
            --distill_p2_gamma=3 \
            --distill_feats=False \
            --distill_feats_lambda=0.0 \
            --distill_agnostic=True \
            --distill_agnostic_gamma=50 \
            --distill_agnostic_lambda=0.1 \
            --save_interval 5000 \
            --last_iter=80k \
            --gpu $gpu


# additional

source_dir='/mnt/raid/aahq_expressiveness_man_noglasses'
epochs="010000 020000 040000 060000 080000"
gpu=${gpu:0:1}

for epoch in $epochs
do
    log_dir='/home/jiwan.hur/P2-weighting/logs/'$exp'/model'$epoch'.pt'
    sample_dir='/home/jiwan.hur/samples/'$exp'_'$epoch'/images'
    bash sampling.sh $gpu ./logs/$exp/model$epoch.pt $sample_dir
    python metric.py --gpu $gpu \
                    --source_dir $source_dir \
                    --sample_dir $sample_dir \
                    --use_dataparallel True
done