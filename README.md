# SDFT (WACV 2024)

This is the codebase for [Expanding Expressiveness of Diffusion Models with Limited Data via Self-Distillation based Fine-Tuning](https://arxiv.org/abs/2311.01018).

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion) and [P2-Weighting](https://github.com/jychoi118/P2-weighting).

SDFT aims to enhance the expressiveness of the diffusion models trained with limited datasets, which tend to have less diverse and biased attributes. The limited expressiveness not only hampers the generation capability of the model but also results in unsatisfactory outputs in various downstream tasks, such as domain translation and text-guided image manupulation.

<p align="center">
    <img src="https://github.com/user-attachments/assets/0ceb8bfd-6667-46b4-8734-8e80a41a8ca7" width="85%">
</p>

## Pre-trained models

All models are trained at 256x256 resolution.

We use pre-trained FFHQ model from [P2-Weighting](https://github.com/jychoi118/P2-weighting) repository.

Here are the models trained on MetFaces with SDFT: [link](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH).
We obtained the reported values with fine-tuning 10k iterations. The domain-specific feature extractor for EGSDE is also attached. We follow the official implementation of [EGSDE](https://github.com/ML-GSAI/EGSDE) to train the domain-specific feature extractor.

## Requirements
We trained the model on PyTorch 1.7.1, 8 RTX 2080 Ti GPUs.

## Sampling from pre-trained models

### Unconditional Generation

First, set PYTHONPATH variable to point to the root of the repository. Do the same when training new models. 

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Put model checkpoints into a folder `./models/`.

Samples will be saved in `./samples/`.

```
python scripts/image_sample.py --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 1 --num_head_channels 64 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing ddim40 --use_ddim True --model_path models/metface_distill.pt --sample_dir samples
```

We adopt 40 step DDIM sampling for the default config for the efficiency. One can change the sampling strategy by modifying `--timestep_respacing` and `--use_ddim`.

### Domain Translation
We implement the [SDEdit](https://github.com/ermongroup/SDEdit) in `./notebooks/SDEdit.ipynb` and [EGSDE](https://github.com/ML-GSAI/EGSDE) in in `./notebooks/EGSDE.ipynb` for diffusion-based domain translation. For EGSDE, please prepare the `face2portrain.pt` in `./models/` for domain-specific feature extractor.

### Text-Guided Image Manipulation
We follow [Asyrp](https://github.com/kwonminki/Asyrp_official) for the implementation of text-guided image manipulation. 

## Training your models

For MetFaces dataset,
- set `--distill_lambda=0.1` and `--distill_p2_gamma=3` for distillation loss in equation (3) of the paper.
- set `--distill_agnostic=True`, `--distill_agnostic_lambda=0.1` and `--distill_agnostic_gamma=50`  for auxiliary loss in equation (5) of the paper.

Logs and models will be saved in `logs/`. You should modify `--data_dir`. 

```
bash train_ddp_distill.sh
```


