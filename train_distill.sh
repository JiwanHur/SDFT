#!/bin/bash

# gpu="0,1,2,3"
gpu="1"
sh_file_name="train_distill.sh"

l=0.3
g=3
la=0.3
ga=50

exp="aahq_limited1400_distill_p2_"$l"_"$g"_aux_"$la"_"$ga
# exp="aahq_limited_distill_p2_"$l"_"$g"_aux_"$la"_"$ga

export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=$gpu python scripts/image_distill_train.py \
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
            --resume_checkpoint models/ffhq_p2.pt \
            --dist_checkpoint models/ffhq_p2.pt \
            --distill_lambda=$l \
            --distill_p2_gamma=$g \
            --distill_feats=False \
            --distill_feats_lambda=0.05 \
            --distill_agnostic=True \
            --distill_agnostic_gamma=$ga \
            --distill_agnostic_lambda=$la \
            --save_interval 5000 \
            --last_iter=40k \
            --gpu $gpu

        
# additional
source_dir='/mnt/raid/aahq_expressiveness_man_noglasses/'
epochs="010000 020000 030000 040000"

for epoch in $epochs
do
    sample_dir='/home/jiwan.hur/samples_aahq/'$exp'_'$epoch'/images'
    bash sampling.sh $gpu ./logs/$exp/model$epoch.pt $sample_dir
    python metric.py --gpu $gpu \
                    --source_dir $source_dir \
                    --sample_dir $sample_dir \
                    --use_dataparallel False
done