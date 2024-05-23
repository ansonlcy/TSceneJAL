import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt
import random

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_al_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from al_train_utils.al_train_utils import al_loop_train

import tqdm
from pcdet.models import load_data_to_gpu
import numpy as np


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoints to start from')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max number of saved checkpoints')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--rules', type=int, default=1, help='the rules when search')
    parser.add_argument('--rules_2', type=int, default=0, help='the rules when use difficulty search')
    parser.add_argument('--score_thresh', type=float, default=0.3, help='')
    parser.add_argument('--eu_theta', type=float, default=1, help='the theta of theta*eu')
    parser.add_argument('--au_theta', type=float, default=1, help='the theta of theta*au')
    parser.add_argument('--score_plus', action='store_true', default=False, help='')
    parser.add_argument('--score_reverse', action='store_true', default=False, help='')
    parser.add_argument('--consider_other', action='store_true', default=True, help='consider other type(pedestrain,cyclist)')
    parser.add_argument('--group_name', type=str, default='al_group', help='group name')

    # total_loop=7, init_set_len=200, search_num_each=200, loop_epochs=50, epoch_step=2
    parser.add_argument('--total_loop', type=int, default=7, help='total loop')
    parser.add_argument('--init_set_len', type=int, default=200, help='init set len')
    parser.add_argument('--search_num_each', type=int, default=200, help='search num each')
    parser.add_argument('--loop_epochs', type=int, default=50, help='loop epochs')
    parser.add_argument('--epoch_step', type=int, default=2, help='epoch step')

    parser.add_argument('--k1', type=float, default=3, help='k1')
    parser.add_argument('--k2', type=float, default=2.5, help='k2')


    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    torch.set_num_threads(1)
    args, cfg = parse_config()
    dist_train = False

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.group_name /args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    stat_k = None
    if args.rules == 12:
        stat_k = cfg.ACTIVE_TRAIN.STAT_K


    # kitti config
    al_loop_train(cfg, args, logger, total_loop=5, init_set_len=200, search_num_each=200, loop_epochs=50, epoch_step=2,
                  stat_k=stat_k, k1=3, k2=2.5)

    # fasthest sampling test
    # al_loop_train(cfg, args, logger, total_loop=1, init_set_len=200, search_num_each=200, loop_epochs=50, epoch_step=2,
    #               stat_k=stat_k, k1=3, k2=2.5)


    # more data test
    # al_loop_train(cfg, args, logger, total_loop=17, init_set_len=200, search_num_each=200, loop_epochs=50, epoch_step=2,
    #               stat_k=stat_k, k1=3, k2=2.5)

    # al_loop_train(cfg, args, logger, total_loop=7, init_set_len=200, search_num_each=200, loop_epochs=50, epoch_step=2,
    #               stat_k=stat_k, k1=2)

    # lyft config
    # al_loop_train(cfg, args, logger, total_loop=5, init_set_len=400, search_num_each=400, loop_epochs=10, epoch_step=2,
    #               stat_k=stat_k, k1=3, k2=2)

    # budgest test and k test in kitti
    # al_loop_train(cfg, args, logger, total_loop=args.total_loop, init_set_len=args.init_set_len, search_num_each=args.search_num_each, loop_epochs=args.loop_epochs, epoch_step=args.epoch_step,
    #               stat_k=stat_k, k1=args.k1, k2=args.k2)


if __name__ == '__main__':
    main()
