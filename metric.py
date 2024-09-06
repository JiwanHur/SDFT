from cleanfid import fid
from guided_diffusion import logger
import argparse
import datetime
import pdb
import glob
from guided_diffusion.script_util import add_dict_to_argparser


def main():
    args = create_argparser().parse_args()

    device=f'cuda:{args.gpu}'
    use_dataparallel=args.use_dataparallel
    suffix = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M")

    if args.source_dir:
        log_dir = args.sample_dir.split('images')[0]
        logger.configure(dir=log_dir, log_suffix=suffix)
        
        f1 = args.source_dir
        f2 = args.sample_dir
        logger.log(f'source_dir: {f1}, target_dir: {f2}')
        score = fid.compute_fid(f1, f2, device=device, use_dataparallel=use_dataparallel)
        logger.log(f'fid: {score}')
        score = fid.compute_kid(f1, f2, device=device, use_dataparallel=use_dataparallel)
        logger.log(f'kid(x10^3): {score*(10**3)}')
        
    else: # args.source_dir does not exist:
        # fdir1 = ['../dataset/afhqcat_traintest/']
        # fdir1 = ['../dataset/metface/256/']
        # fdir1 = ['/mnt/raid/aahq_expressiveness_man_noglasses']
        fdir1 = ['/mnt/raid/aahq_256']

        fdir2 = ['/home/jiwan.hur/samples/aahq_limited_transfer_010000/images',
                 '/home/jiwan.hur/samples/aahq_limited_transfer_020000/images',
                 '/home/jiwan.hur/samples/aahq_limited_transfer_030000/images',
                 '/home/jiwan.hur/samples/aahq_limited_transfer_040000/images',
                 '/home/jiwan.hur/samples/aahq_limited_transfer_060000/images',
                 '/home/jiwan.hur/samples/aahq_limited_transfer_080000/images',]
        # fdir2 = sorted(glob.glob('../samples/afhqcat_gw_distill_4_*'))
        # fdir2 = sorted(glob.glob('../samples/afhqcat_gw_distill_5_*'))
        # fdir2 = sorted(glob.glob('../samples/afhqcat_gw_distill_13_*'))
        # fdir2 = sorted(glob.glob('../i2i_quant/ILVR_met/*'))

        for f1 in fdir1:
            for f2 in fdir2:
                logger.configure(dir=f2, log_suffix=suffix)
                try:
                    logger.log(f2)
                    score = fid.compute_fid(f1, f2, device=device, use_dataparallel=use_dataparallel)
                    logger.log(f'fid: {score}')
                    score = fid.compute_kid(f1, f2, device=device, use_dataparallel=use_dataparallel)
                    logger.log(f'kid(x10^3): {score*(10**3)}')
                except:
                    logger.log(f'{f2} is empty')

        # fdir3 = ['../dataset/metface/256/']
        # fdir4 = ['../samples/metface_distill_14_10k_2',]
                
        # for f1 in fdir3:
        #     for f2 in fdir4:
        #         logger.log(f2)
        #         score = fid.compute_fid(f1, f2, device=device, use_dataparallel=use_dataparallel)
        #         logger.log(f'fid: {score}')
        #         score = fid.compute_kid(f1, f2, device=device, use_dataparallel=use_dataparallel)
        #         logger.log(f'kid(x10^3): {score*(10**3)}')

def create_argparser():
    defaults = dict(
        gpu="0",
        source_dir="",
        sample_dir="",
        use_dataparallel=False,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__=='__main__':
    main()