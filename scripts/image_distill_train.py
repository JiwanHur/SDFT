"""
Train a diffusion model on images.
"""

import argparse
import shutil
import os
import copy
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util_distill import TrainLoop


def main():
    parser, distill_keys = create_argparser()
    args = parser.parse_args()
    
    dist_util.setup_dist(gpu=args.gpu)
    logger.configure(dir=args.log_dir)
    shutil.copy(args.sh_file_name, os.path.join(args.log_dir, args.sh_file_name))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    model_t = copy.deepcopy(model).eval()
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("args...")
    logger.log(args)
    
    logger.log("training...")
    TrainLoop(
        model=model,
        model_t=model_t,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        dist_checkpoint=args.dist_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        log_dir=args.log_dir,
        distill_kwargs=args_to_dict(args, distill_keys)
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        log_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        dist_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        sh_file_name="",
        gpu="",
    )
    distill_defaults = dict(
        distill_lambda=0.1,
        distill_p2_gamma=3,
        distill_feats=False,
        distill_feats_lambda=0.05,
        distill_agnostic=False, # domain-agnostic noise distill
        distill_agnostic_gamma=100,
        distill_agnostic_lambda=0.1,
        last_iter='200k',
        distill_min_snr=False,
    )
    defaults.update(distill_defaults)
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser, distill_defaults.keys()


if __name__ == "__main__":
    main()
