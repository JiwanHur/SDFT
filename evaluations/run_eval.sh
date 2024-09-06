#!/bin/bash

real_dir="/mnt/afhq/train/samples_dog.npz"
sample_dir="../samples/afhqdog_step_40/samples_10000x256x256x3.npz"
python evaluator.py $real_dir $sample_dir