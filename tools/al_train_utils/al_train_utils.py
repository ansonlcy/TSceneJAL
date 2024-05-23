import glob
import os
from pathlib import Path
import tqdm
import numpy as np
import torch
import random
import time
import re
from torch.nn.utils import clip_grad_norm_
from pcdet.datasets import build_al_dataloader
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils, commu_utils
from pcdet.models import load_data_to_gpu, build_network, model_fn_decorator
from .optimization import build_optimizer, build_scheduler
from tensorboardX import SummaryWriter
from pcdet.query_strategies.mdn_change_data_pool import mdn_strategies
from pcdet.query_strategies.common_change_data_pool import random_change_data_pool
from pcdet.query_strategies.multi_stage_change_data_pool import multi_stage_strategies
from pcdet.query_strategies.stat_mdn_change_data_pool import stat_mdn_strategies
from pcdet.query_strategies.badge_change_data_pool import badge_strategies
from pcdet.query_strategies.coreset_change_data_pool import coreset_strategies
from pcdet.query_strategies.crb_change_data_pool import crb_strategies


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()

    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        loss, tb_dict, disp_dict = model_func(model, batch)

        # loss太大的话直接跳过
        flag_1 = False
        if loss > 10e4:
            flag_1 = True
            print("**********loss**********")

        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)

        flag_2 = False
        for p in model.parameters():
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                flag_2 = True
                print("**********JUMP************")
                break
            # break

        if not flag_2 and not flag_1:
            optimizer.step()

        accumulated_iter += 1

        cur_batch_time = time.time() - end
        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})',
                'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def al_train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                   start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, loop_step, train_sampler=None,
                   lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc=f'loop:{loop_step} epochs', dynamic_ncols=True,
                     leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / f'checkpoint_loop_{loop_step}_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_loop_%d_epoch_%d' % (loop_step, trained_epoch))
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoints'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)


def save_pool(label_pool, unlabel_pool, dir):
    save_dict = {'label_pool': label_pool, 'unlabel_pool': unlabel_pool}
    np.save(dir, save_dict)


def al_loop_train(cfg, args, logger, total_loop, init_set_len=1000,
                  search_num_each=500, loop_epochs=100, epoch_step=2, k1=3, k2=2):
    # first use the init data to train the init model

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.group_name / args.extra_tag
    ckpt_root_dir = output_dir / 'ckpt'
    tensorboard_root_dir = output_dir / 'tensorboard'
    pool_dir = output_dir / 'pool'
    tensorboard_root_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root_dir.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(parents=True, exist_ok=True)

    cur_loop = 0

    continue_train = False
    # next is to check if train from the exist ckpt
    ckpt_dir_list = glob.glob(str(ckpt_root_dir / '*'))
    if len(ckpt_dir_list) != 0:
        ckpt_dir_list.sort(key=lambda x: x.split('_')[-1])
        cur_loop = int(ckpt_dir_list[-1].split('_')[-1])
        if len(glob.glob(ckpt_dir_list[-1] + "/*.pth")) == 0:
            cur_loop -= 1
        continue_train = True

    # create lyft dataset to get the total nums of samples
    tmp_train_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers,
        logger=logger,
        training=True,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )
    num_total_samples = len(tmp_train_set.kitti_infos)
    logger.info(f'num_total_samples: {num_total_samples}')
    del tmp_train_set

    if not continue_train:  # if is the init train loop
        ckpt_dir = ckpt_root_dir / f'loop_{cur_loop}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb_log = SummaryWriter(log_dir=str(tensorboard_root_dir / f'tensorboard_loop_{cur_loop}'))

        init_set = list(range(num_total_samples))
        random.shuffle(init_set)
        label_set = init_set[:init_set_len]
        unlabel_set = init_set[init_set_len:]
        save_pool(label_set, unlabel_set, pool_dir / f'pool_loop_{cur_loop}.npy')

        label_dataset, label_dataloader, unlabel_dataset, unlabel_dataloader = build_al_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            slice_set=(label_set, unlabel_set),
            seed=666 if args.fix_random_seed else None,
            workers=args.workers
        )

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=label_dataset)
        model.cuda()
        optimizer = build_optimizer(model, cfg.OPTIMIZATION)

        start_epoch = it = 0
        last_epoch = -1

        model.train()  # before wrap to DistributedDataParallel to support fixed some parameters

        lr_scheduler, lr_warmup_scheduler = build_scheduler(
            optimizer, total_iters_each_epoch=len(label_dataloader), total_epochs=loop_epochs,
            last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
        )
        logger.info(
            '**********************Start Active init learning training. Samples:%s********************** ' % init_set_len)

        al_train_model(model,
                       optimizer,
                       label_dataloader,
                       model_func=model_fn_decorator(),
                       lr_scheduler=lr_scheduler,
                       optim_cfg=cfg.OPTIMIZATION,
                       start_epoch=start_epoch,
                       total_epochs=loop_epochs,
                       start_iter=it,
                       rank=0,
                       loop_step=cur_loop,
                       tb_log=tb_log,
                       ckpt_save_dir=ckpt_dir,
                       lr_warmup_scheduler=lr_warmup_scheduler,
                       ckpt_save_interval=args.ckpt_save_interval,
                       max_ckpt_save_num=args.max_ckpt_save_num)

        cur_loop += 1

    while cur_loop <= total_loop:

        ckpt_dir = ckpt_root_dir / f'loop_{cur_loop}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb_log = SummaryWriter(log_dir=str(tensorboard_root_dir / f'tensorboard_loop_{cur_loop}'))

        need_search = not (pool_dir / f'pool_loop_{cur_loop}.npy').is_file()
        if continue_train:
            if not need_search:
                pool_dict = np.load(pool_dir / f'pool_loop_{cur_loop}.npy', allow_pickle=True).item()
            else:
                pool_dict = np.load(pool_dir / f'pool_loop_{cur_loop - 1}.npy', allow_pickle=True).item()
            label_set = pool_dict['label_pool']
            unlabel_set = pool_dict['unlabel_pool']
            label_dataset, label_dataloader, unlabel_dataset, unlabel_dataloader = build_al_dataloader(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                batch_size=args.batch_size,
                slice_set=(label_set, unlabel_set),
                seed=666 if args.fix_random_seed else None,
                workers=args.workers
            )

        if not continue_train or need_search:
            ckpt_list = glob.glob(
                str(ckpt_root_dir / f'loop_{cur_loop - 1}' / f'*checkpoint_loop_{cur_loop - 1}_epoch_*.pth'))
            if len(ckpt_list) > 0:
                ckpt_list.sort(key=lambda x: int(re.search(r'_epoch_(\d+)\.pth', str(x)).group(1)))
            if args.rules not in [0, 13, 14, 15, 16, 17, 152, 153, 20, 21, 22]:
                label_set, unlabel_set, _ = mdn_strategies(cfg, args, model, ckpt_list[-1], logger,
                                                           (label_set, unlabel_set),
                                                           search_num_each, rules=args.rules,
                                                           score_thresh=args.score_thresh, eu_theta=args.eu_theta,
                                                           au_theta=args.au_theta,
                                                           score_plus=args.score_plus,
                                                           score_reverse=args.score_reverse,
                                                           consider_other=args.consider_other)

            elif args.rules in [13, 14, 15, 16, 17, 152, 153]:
                label_set, unlabel_set, _ = multi_stage_strategies(cfg, args, model, ckpt_list[-1], logger,
                                                                   (label_set, unlabel_set),
                                                                   search_num_each, rules=args.rules,
                                                                   score_thresh=args.score_thresh,
                                                                   eu_theta=args.eu_theta,
                                                                   au_theta=args.au_theta,
                                                                   k1=k1, k2=k2)
            elif args.rules == 20:
                label_set, unlabel_set, _ = badge_strategies(cfg, args, model, ckpt_list[-1], logger,
                                                             (label_set, unlabel_set), search_num_each)

            elif args.rules == 21:
                label_set, unlabel_set, _ = coreset_strategies(cfg, args, model, ckpt_list[-1], logger,
                                                               (label_set, unlabel_set), search_num_each,
                                                               feature_save_dir=None)
            elif args.rules == 22:
                label_set, unlabel_set, _ = crb_strategies(cfg, args, model, ckpt_list[-1], logger,
                                                           (label_set, unlabel_set), search_num_each, k1=k1, k2=k2)

            elif args.rules == 0:
                label_set, unlabel_set = random_change_data_pool(logger, (label_set, unlabel_set), search_num_each)

            else:
                raise NotImplementedError

            save_pool(label_set, unlabel_set, pool_dir / f'pool_loop_{cur_loop}.npy')

        label_dataset, label_dataloader, unlabel_dataset, unlabel_dataloader = build_al_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            slice_set=(label_set, unlabel_set),
            seed=666 if args.fix_random_seed else None,
            workers=args.workers
        )

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=label_dataset)
        model.cuda()
        optimizer = build_optimizer(model, cfg.OPTIMIZATION)

        start_epoch = it = 0
        last_epoch = -1

        if continue_train:
            loop_epochs = loop_epochs + (cur_loop - 1) * epoch_step
            loop_epochs = min(loop_epochs, 80)
            ckpt_list = glob.glob(
                str(ckpt_root_dir / f'loop_{cur_loop}' / f'*checkpoint_loop_{cur_loop}_epoch_*.pth'))
            if len(ckpt_list) > 0:
                ckpt_list.sort(key=lambda x: int(re.search(r'_epoch_(\d+)\.pth', str(x)).group(1)))
                it, start_epoch = model.load_params_with_optimizer(ckpt_list[-1], optimizer=optimizer, logger=logger)
                last_epoch = start_epoch + 1

        model.train()

        lr_scheduler, lr_warmup_scheduler = build_scheduler(
            optimizer, total_iters_each_epoch=len(label_dataloader), total_epochs=loop_epochs,
            last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
        )
        logger.info(
            '**********************Active learning training loop:%s Samples:%s********************** ' % (
                cur_loop, len(label_set)))

        al_train_model(model,
                       optimizer,
                       label_dataloader,
                       model_func=model_fn_decorator(),
                       lr_scheduler=lr_scheduler,
                       optim_cfg=cfg.OPTIMIZATION,
                       start_epoch=start_epoch,
                       total_epochs=loop_epochs,
                       start_iter=it,
                       rank=0,
                       loop_step=cur_loop,
                       tb_log=tb_log,
                       ckpt_save_dir=ckpt_dir,
                       lr_warmup_scheduler=lr_warmup_scheduler,
                       ckpt_save_interval=args.ckpt_save_interval,
                       max_ckpt_save_num=args.max_ckpt_save_num)

        loop_epochs += epoch_step
        loop_epochs = min(loop_epochs, 80)
        cur_loop += 1
        continue_train = False


def al_loop_train_by_pp(cfg, args, logger, total_loop, copy_pool_root):
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root_dir = output_dir / 'ckpt'
    ckpt_root_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_root_dir = output_dir / 'tensorboard'
    tensorboard_root_dir.mkdir(parents=True, exist_ok=True)

    loop_epochs_dict = {0: 50, 1: 50, 2: 52, 3: 54, 4: 56, 5: 58, 6: 60, 7: 62}
    for loop in range(0, total_loop + 1):
        ckpt_dir = ckpt_root_dir / f'loop_{loop}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb_log = SummaryWriter(log_dir=str(tensorboard_root_dir / f'tensorboard_loop_{loop}'))

        # pool_dir = cfg.ROOT_DIR / 'output' / 'kitti_models' / 'pointpillar_mdn' / 'pointpillar_al_test' / args.pool_name / 'ckpt' / f'pool_loop_{loop}.npy'
        # pool_dir = cfg.ROOT_DIR / 'output' / 'kitti_models' / 'pp_corner_al' / 'c2_v1_down_mix_al' / 'test_7_25' / args.pool_name / 'pool' / f'pool_loop_{loop}.npy'

        pool_dir = Path(copy_pool_root) / f'pool_loop_{loop}.npy'
        _loop = np.load(pool_dir, allow_pickle=True).item()
        label_set = _loop['label_pool']
        unlabel_set = _loop['unlabel_pool']

        label_dataset, label_dataloader, unlabel_dataset, unlabel_dataloader = build_al_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            slice_set=(label_set, unlabel_set),
            seed=666 if args.fix_random_seed else None,
            workers=args.workers
        )

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=label_dataset)
        model.cuda()
        optimizer = build_optimizer(model, cfg.OPTIMIZATION)

        start_epoch = it = 0
        last_epoch = -1

        model.train()  # before wrap to DistributedDataParallel to support fixed some parameters

        lr_scheduler, lr_warmup_scheduler = build_scheduler(
            optimizer, total_iters_each_epoch=len(label_dataloader), total_epochs=args.epochs,
            last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
        )
        logger.info(
            '**********************Start Active learning training loop %s. Samples:%s********************** ' % (
                loop, len(label_set)))

        al_train_model(model,
                       optimizer,
                       label_dataloader,
                       model_func=model_fn_decorator(),
                       lr_scheduler=lr_scheduler,
                       optim_cfg=cfg.OPTIMIZATION,
                       start_epoch=start_epoch,
                       total_epochs=loop_epochs_dict[loop],
                       start_iter=it,
                       rank=0,
                       loop_step=loop,
                       tb_log=tb_log,
                       ckpt_save_dir=ckpt_dir,
                       lr_warmup_scheduler=lr_warmup_scheduler,
                       ckpt_save_interval=args.ckpt_save_interval,
                       max_ckpt_save_num=args.max_ckpt_save_num)


def al_single_loop_train(cfg, args, logger):
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.group_name / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root_dir = output_dir / 'ckpt'
    ckpt_root_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_root_dir = output_dir / 'tensorboard'
    tensorboard_root_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ckpt_root_dir / f'loop_{args.loop_num}_fix'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_log = SummaryWriter(log_dir=str(tensorboard_root_dir / f'tensorboard_loop_{args.loop_num}_fix'))

    loop_set_dir = output_dir / 'pool' / f'pool_loop_{args.loop_num}.npy'
    pool_dict = np.load(loop_set_dir, allow_pickle=True).item()
    label_set = pool_dict['label_pool']
    unlabel_set = pool_dict['unlabel_pool']
    print(f"Train the single loop {args.loop_num}. {len(label_set)} labels and {len(unlabel_set)} unlabels.")

    label_dataset, label_dataloader, unlabel_dataset, unlabel_dataloader = build_al_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        slice_set=(label_set, unlabel_set),
        seed=666 if args.fix_random_seed else None,
        workers=args.workers
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=label_dataset)
    model.cuda()
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    start_epoch = it = 0
    last_epoch = -1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(label_dataloader), total_epochs=args.loop_epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )
    logger.info(
        '**********************Start Active learning training loop %s. Samples:%s********************** ' % (
            args.loop_num, len(label_set)))

    al_train_model(model,
                   optimizer,
                   label_dataloader,
                   model_func=model_fn_decorator(),
                   lr_scheduler=lr_scheduler,
                   optim_cfg=cfg.OPTIMIZATION,
                   start_epoch=start_epoch,
                   total_epochs=args.loop_epochs,
                   start_iter=it,
                   rank=0,
                   loop_step=args.loop_num,
                   tb_log=tb_log,
                   ckpt_save_dir=ckpt_dir,
                   lr_warmup_scheduler=lr_warmup_scheduler,
                   ckpt_save_interval=args.ckpt_save_interval,
                   max_ckpt_save_num=args.max_ckpt_save_num)


if __name__ == "__main__":
    import argparse
    from pcdet.config import cfg, cfg_from_yaml_file
    import os
    from pcdet.utils import common_utils

    common_utils.set_random_seed(666)
    os.chdir('../')

    parser = argparse.ArgumentParser(description='arg parser')
    args = parser.parse_args()
    args.batch_size = 1
    args.workers = 0
    args.fix_random_seed = True
    args.rules = 15
    args.rules_2 = 2
    args.score_thresh = 0.3
    args.au_theta = 0.0

    cfg_dir = 'cfgs/kitti_models/pp_box_mdn/pointpillar_box_mdn_com3.yaml'
    # cfg_dir = 'cfgs/kitti_models/pp_box_mdn/pointpillar_box_mdn_com3_mc_2_0_3.yaml'
    # cfg_dir = 'cfgs/lyft_models/pp_box_mdn/pointpillar_box_mdn_com3_mc.yaml'
    # cfg_dir = 'cfgs/kitti_models/pp_corner_al/c2_v1_down_mix_mc_car_al.yaml'
    # ckpt = '../output_cse/kitti_models/pointpillar_mdn/pointpillar_mdn_com3_g0.5/pp_mdn_com3_g0.5/ckpt/checkpoint_epoch_120.pth'
    # ckpt = '../checkpoints/checkpoint_loop_0_epoch_100.pth'

    # ckpt = '/home/leicy/projects/openpcdet/checkpoints/pp_box_mdn_com3_mc_2.pth'
    # ckpt = '/home/leicy/projects/openpcdet/checkpoints/pp_box_mdn_lyft.pth'
    ckpt = '/home/leicy/projects/openpcdet/checkpoints/checkpoint_loop_3_epoch_54.pth'
    cfg_from_yaml_file(cfg_dir, cfg)
    logger = common_utils.create_logger('./log.txt', rank=cfg.LOCAL_RANK)

    # pool_dir = '../output/kitti_models/pp_corner_al/c2_v1_down_sep_mix_al/test_7_24/al_train_test_corner/pool/pool_loop_0.npy'
    # pool_dir = 'al_colloc/al_data_pool/test_22_8_9/au_eu/pool_loop_0.npy'
    pool_dir = 'al_colloc/al_data_pool/pool_loop_3.npy'
    pool = np.load(pool_dir, allow_pickle=True).item()
    # label_pool = pool['label_pool']
    # unlabel_pool = pool['unlabel_pool']
    label_pool = list(range(20))
    unlabel_pool = list(range(20, 70))

    # unlabel_pool = [i for i in range(3712)]

    label_dataset, label_dataloader, unlabel_dataset, unlabel_dataloader = build_al_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        slice_set=([], unlabel_pool),
        seed=666,
        workers=0
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=label_dataset)
    model.cuda()
    model.eval()

    label_set, unlabel_set, choose_id = multi_stage_strategies(cfg, args, model, ckpt, logger,
                                                               (label_pool, unlabel_pool),
                                                               20, rules=args.rules, score_thresh=args.score_thresh,
                                                               au_theta=args.au_theta, k1=2)

    # label_set, unlabel_set, choose_id = crb_strategies(cfg, args, model, ckpt, logger, (label_pool, unlabel_pool), search_num_each=20, k1=3, k2=2)
    # label_set, unlabel_set, choose_id = coreset_strategies(cfg, args, model, ckpt, logger, (label_pool, unlabel_pool),
    #                                                    search_num_each=20)

    # label_set, unlabel_set = change_data_pool_by_difficulty(cfg, args, logger, (label_pool, unlabel_pool), 100)

    # idx_to_id = {i:unlabel_dataset.kitti_infos[i]['image']['image_idx'] for i in
    #              range(len(unlabel_dataset.kitti_infos))}
    # l = [idx_to_id[i] for i in [1987, 710, 1679, 3040, 1361, 1055, 213, 656, 3632, 993]]
    # print(l)

    # ['000934' '005183' '003957' '003454' '003297' '005896' '000547' '000891'
    #  '006617' '006006']
