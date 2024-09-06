#!/bin/bash
gpu='3'
source_dir='/mnt/raid/aahq_256'
# source_dir='/mnt/raid/metface/256'

# exp='metface_ablation_distill_minsnr_5'
exp='aahq_limited1400_distill_p2_0.2_3_aux_0.2_50'
# epochs="078000 076000 074000 072000 070000 068000 066000 064000 062000 "
# epochs="010000 020000 030000 040000 050000 060000 070000 080000"
epochs="040000"

for epoch in $epochs
do
    log_dir='/home/jiwan.hur/ftp_home/diffusion/P2-weighting/logs/'$exp'/model'$epoch'.pt'
    sample_dir='/home/jiwan.hur/samples/'$exp'_'$epoch'/images'
    bash sampling.sh $gpu ./logs/$exp/model$epoch.pt $sample_dir
    python metric.py --gpu $gpu \
                    --source_dir $source_dir \
                    --sample_dir $sample_dir \
                    --use_dataparallel False
done


