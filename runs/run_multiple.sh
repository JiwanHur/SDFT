#!/bin/bash

# distill_lambda, distill_agnostic_lambda, distill_feats
bash runs/train_ddp_distill_sub.sh logs/metface_distill_9 0.1 1.0 False
bash runs/train_ddp_distill_sub.sh logs/metface_distill_10 0.1 1.0 True
bash runs/train_ddp_distill_sub.sh logs/metface_distill_11 0.2 0.5 False
bash runs/train_ddp_distill_sub.sh logs/metface_distill_12 0.2 0.5 True