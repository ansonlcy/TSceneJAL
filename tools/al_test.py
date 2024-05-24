import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoints to start from')
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--root_dir', type=str, default=None, help='The root dir when eval')
    parser.add_argument('--start_loop', type=int, default=0, help='The start eval loop')
    parser.add_argument('--end_loop', type=int, default=5, help='The end of tht eval loop')
    parser.add_argument('--eval_num_per_loop', type=int, default=1, help='The number of epochs on each loop eval')
    parser.add_argument('--fix_loop', action='store_true', default=False, help='if we use single loop train')


    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'


    return args, cfg


def repeat_eval_ckpt(model, test_loader, args, logger, ckpt_dir, start_loop=0, end_loop=6, eval_num_per_loop=10, fix_loop=False):
    eval_out_dir = ckpt_dir.parent / 'eval'
    tensorboard_dir = eval_out_dir / 'tensorboard_log'
    if fix_loop:
        path_last = "_fix"
    else:
        path_last = ""
    for loop in range(start_loop, end_loop + 1):

        tb_log = SummaryWriter(log_dir=str(tensorboard_dir / f'tensorboard_loop_{loop}{path_last}'))

        ckpt_list = glob.glob(str(ckpt_dir / f'loop_{loop}{path_last}' / f'checkpoint_loop_{loop}_epoch_*.pth'))

        ckpt_list.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))

        cur_ckpt_list = ckpt_list[-eval_num_per_loop:]

        for cur_ckpt in cur_ckpt_list:
            cur_epoch_id = int(re.findall(r'\d+', cur_ckpt)[-1])
            model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=False)
            model.cuda()
            # start evaluation
            cur_result_dir = eval_out_dir / f'loop_{loop}{path_last}' / ('epoch_%s' % cur_epoch_id)
            tb_dict = eval_utils.eval_one_epoch(
                cfg, model, test_loader, cur_epoch_id, logger,
                result_dir=cur_result_dir, save_to_file=args.save_to_file
            )
            # record this epoch which has been evaluated
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

            logger.info('Loop %s, Epoch %s has been evaluated' % (loop, cur_epoch_id))


def main():
    torch.set_num_threads(1)
    args, cfg = parse_config()
    root_dir = Path(args.root_dir)
    eval_output_dir = root_dir / 'eval'
    tensorboard_dir = eval_output_dir / 'tensorboard_log'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    ckpt_dir = root_dir / 'ckpt'
    repeat_eval_ckpt(model, test_loader, args, logger, ckpt_dir, start_loop=args.start_loop, end_loop=args.end_loop, eval_num_per_loop=args.eval_num_per_loop, fix_loop=args.fix_loop)


if __name__ == '__main__':
    main()
