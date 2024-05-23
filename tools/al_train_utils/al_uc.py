import glob
import os

import tqdm
import numpy as np
import torch
import random
import time
from torch.nn.utils import clip_grad_norm_
from pcdet.datasets import build_al_dataloader
from pcdet.utils import common_utils, commu_utils
from pcdet.models import load_data_to_gpu, build_network, model_fn_decorator
from optimization import build_optimizer, build_scheduler
from tensorboardX import SummaryWriter


def random_change_data_pool(slice_set, search_num_each):
    (label_pool, unlabel_pool) = slice_set
    random.shuffle(label_pool)
    random.shuffle(unlabel_pool)
    choose = unlabel_pool[:search_num_each]

    label_pool = list(set(label_pool) | (set(choose)))
    unlabel_pool = list(set(unlabel_pool) - set(choose))

    return label_pool, unlabel_pool


def change_data_pool(cfg, args, model, ckpt_dir, logger, slice_set, search_num_each, rules=1, score_thresh=0.3,
                     score_plus=False, score_reverse=False):
    (label_pool, unlabel_pool) = slice_set

    _, _, unlabel_dataset, unlabel_dataloader = build_al_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        slice_set=slice_set,
        seed=666 if args.fix_random_seed else None,
        workers=args.workers
    )

    logger.info('**********************Start search:label pool:%s  unlabel_pool:%s**********************' %
                (len(label_pool), len(unlabel_pool)))

    model.load_params_from_file(filename=ckpt_dir, logger=logger)
    model.cuda()
    model.eval()

    progress_bar = tqdm.tqdm(total=len(unlabel_dataloader), leave=True, desc='search', dynamic_ncols=True)

    det_annos = []
    for i, batch_dict in enumerate(unlabel_dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

            annos = unlabel_dataloader.dataset.generate_al_prediction_dicts(
                batch_dict, pred_dicts, unlabel_dataset.class_names
            )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.update()

        # if i == 20:
        #     break

    al_uc_list = []
    score_list = []
    id_list = []
    car_num_list = []

    for idx, det in enumerate(det_annos):
        if not 'Car' in det['name'].tolist():
            al_uc_list.append(np.array([]))
            score_list.append(np.array([]))
            car_num_list.append(0)
        else:
            car_idx = np.where(det['name'] == 'Car')[0]
            choose_idx = car_idx[det['score'][car_idx] >= score_thresh]
            al_uc_list.append(np.max(det['al'][choose_idx], axis=1))
            score_list.append(det['score'][choose_idx])
            car_num_list.append(len(choose_idx))
        id_list.append(det['frame_id'])

    mean_al_uc = np.mean([val for sub in al_uc_list for val in sub])
    stdev_al_uc = np.std([val for sub in al_uc_list for val in sub])

    if np.isnan(mean_al_uc) or stdev_al_uc == 0:
        mean_al_uc = 0
        stdev_al_uc = 1

    uc_min = -9999.0

    # for i in range(len(al_uc_list)):
    #     if rules == 1:
    #         d = max(al_uc_list[i])
    #     elif rules == 2:
    #         d = sum(al_uc_list[i])
    #     elif rules == 3:
    #         d = np.mean(al_uc_list[i])
    #     al_uc_list[i] = (d - mean_al_uc) / stdev_al_uc
    if score_reverse:
        for i in range(len(score_list)):
            score_list[i] = 1 - score_list[i]

    if score_plus:
        for i in range(len(al_uc_list)):
            al_uc_list[i] = ((al_uc_list[i] - mean_al_uc) / stdev_al_uc) * score_list[i]
    else:
        for i in range(len(al_uc_list)):
            al_uc_list[i] = (al_uc_list[i] - mean_al_uc) / stdev_al_uc

    for idx, d in enumerate(al_uc_list):
        if d.size == 0:
            al_uc_list[idx] = np.array([uc_min])

    al_uc_max = []
    al_uc_sum = []
    al_uc_mean = []
    al_uc_num = []

    for i in range(len(al_uc_list)):
        al_uc_max.append(max(al_uc_list[i]))
        al_uc_sum.append(sum(al_uc_list[i]))
        al_uc_mean.append(np.mean(al_uc_list[i]))
        al_uc_num.append(car_num_list[i])

    d = {}
    for i in range(len(al_uc_list)):
        d.update({id_list[i]: {
            'max': al_uc_max[i],
            'sum': al_uc_sum[i],
            'mean': al_uc_mean[i],
            'num': al_uc_num[i]
        }})
    np.save('./al_uc_data_2.npy', d)




    return label_pool, unlabel_pool, choose_id


def change_data_pool_by_difficulty(cfg, args, logger, slice_set, search_num_each):
    from collections import Counter

    (label_pool, unlabel_pool) = slice_set

    _, _, unlabel_dataset, unlabel_dataloader = build_al_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        slice_set=slice_set,
        seed=666 if args.fix_random_seed else None,
        workers=args.workers
    )

    id_to_idx = {unlabel_dataset.kitti_infos[i]['image']['image_idx']: i for i in
                 range(len(unlabel_dataset.kitti_infos))}

    easy_num_list = []
    moderate_num_list = []
    hard_num_list = []
    unknown_num_list = []

    for idx in unlabel_pool:
        anno = unlabel_dataset.kitti_infos[idx]['annos']
        car_idx = anno['name'] == 'Car'
        diff_count = Counter(anno['difficulty'][car_idx])
        easy_num_list.append(diff_count[0] if 0 in diff_count.keys() else 0)
        moderate_num_list.append(diff_count[1] if 1 in diff_count.keys() else 0)
        hard_num_list.append(diff_count[2] if 2 in diff_count.keys() else 0)
        unknown_num_list.append(diff_count[-1] if -1 in diff_count.keys() else 0)

    if args.rules_2 == 0:
        sorted_indices = np.argsort(easy_num_list)[::-1]
    elif args.rules_2 == 1:
        sorted_indices = np.argsort(moderate_num_list)[::-1]
    elif args.rules_2 == 2:
        sorted_indices = np.argsort(hard_num_list)[::-1]
    elif args.rules_2 == -1:
        sorted_indices = np.argsort(unknown_num_list)[::-1]

    array_pool = np.array(unlabel_pool)
    choose_idx = array_pool[sorted_indices][:search_num_each]
    label_pool = list(set(label_pool) | (set(choose_idx)))
    unlabel_pool = list(set(unlabel_pool) - set(choose_idx))
    random.shuffle(label_pool)
    random.shuffle(unlabel_pool)

    logger.info('\n**********************End search:label pool:%s  unlabel_pool:%s**********************' %
                (len(label_pool), len(unlabel_pool)))

    return label_pool, unlabel_pool


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

        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)

        flag = False
        for p in model.parameters():
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                flag = True
                print("**********JUMP************")
                break
            # break

        if not flag:
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


def al_loop_train(cfg, args, logger, total_loop, num_total_samples=3712, init_set_len=1000,
                  search_num_each=500, loop_epochs=100):
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
        ckpt_dir_list.sort(key=os.path.getmtime)
        cur_loop = int(ckpt_dir_list[-1].split('_')[-1])
        continue_train = True

    if not continue_train:
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
                label_set = pool_dict['label_pool']
                unlabel_set = pool_dict['unlabel_pool']
            else:
                pool_dict = np.load(pool_dir / f'pool_loop_{cur_loop - 1}.npy', allow_pickle=True).item()
                label_set = pool_dict['label_pool']
                unlabel_set = pool_dict['unlabel_pool']

        if not continue_train or need_search:
            ckpt_list = glob.glob(
                str(ckpt_root_dir / f'loop_{cur_loop - 1}' / f'*checkpoint_loop_{cur_loop - 1}_epoch_*.pth'))
            if len(ckpt_list) > 0:
                ckpt_list.sort(key=os.path.getmtime)
            if args.rules == 1 or args.rules == 2 or args.rules == 3 or args.rules == 4:
                label_set, unlabel_set, _ = change_data_pool(cfg, args, model, ckpt_list[-1], logger,
                                                             (label_set, unlabel_set),
                                                             search_num_each, rules=args.rules,
                                                             score_thresh=args.score_thresh, score_plus=args.score_plus,
                                                             score_reverse=args.score_reverse)

            elif args.rules == 0:
                label_set, unlabel_set = random_change_data_pool((label_set, unlabel_set), search_num_each)
            elif args.rules == 5:
                label_set, unlabel_set = change_data_pool_by_difficulty(cfg, args, logger, (label_set, unlabel_set),
                                                                        search_num_each)

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
            ckpt_list = glob.glob(
                str(ckpt_root_dir / f'loop_{cur_loop}' / f'*checkpoint_loop_{cur_loop}_epoch_*.pth'))
            if len(ckpt_list) > 0:
                ckpt_list.sort(key=os.path.getmtime)
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

        # train_epoch_in_loop += 10
        cur_loop += 1
        continue_train = False


def al_loop_train_by_pp(cfg, args, logger, total_loop):
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root_dir = output_dir / 'ckpt'
    ckpt_root_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_root_dir = output_dir / 'tensorboard'
    tensorboard_root_dir.mkdir(parents=True, exist_ok=True)

    for loop in range(0, total_loop + 1):
        ckpt_dir = ckpt_root_dir / f'loop_{loop}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb_log = SummaryWriter(log_dir=str(tensorboard_root_dir / f'tensorboard_loop_{loop}'))

        pool_dir = cfg.ROOT_DIR / 'output' / 'kitti_models' / 'pointpillar_mdn' / 'pointpillar_al_test' / args.pool_name / 'ckpt' / f'pool_loop_{loop}.npy'
        loop_6 = np.load(pool_dir, allow_pickle=True).item()
        label_set = loop_6['label_pool']
        unlabel_set = loop_6['unlabel_pool']

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
                       total_epochs=args.epochs,
                       start_iter=it,
                       rank=0,
                       loop_step=loop,
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
    args.batch_size = 4
    args.workers = 0
    args.fix_random_seed = True
    args.rules = 3
    args.rules_2 = 2

    cfg_dir = 'cfgs/kitti_models/pointpillar_mdn/pointpillar_al_test_no_db.yaml'
    # ckpt = '../output_cse/kitti_models/pointpillar_mdn/pointpillar_mdn_com3_g0.5/pp_mdn_com3_g0.5/ckpt/checkpoint_epoch_120.pth'
    # ckpt = '../checkpoints/checkpoint_loop_0_epoch_100.pth'
    ckpt = '../output_cse/kitti_models/pointpillar_mdn/pointpillar_mdn_com3_g0.5/pp_mdn_com3_g0.5/ckpt/checkpoint_epoch_120.pth'
    cfg_from_yaml_file(cfg_dir, cfg)
    logger = common_utils.create_logger('./log.txt', rank=cfg.LOCAL_RANK)

    pool_dir = '../output_cse/kitti_models/pointpillar_mdn/pointpillar_al_test_no_db/al_train_mean_thresh_0.3/pool/pool_loop_0.npy'
    pool = np.load(pool_dir, allow_pickle=True).item()
    label_pool = pool['label_pool']
    unlabel_pool = pool['unlabel_pool']

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
    #
    label_set, unlabel_set, choose_id = change_data_pool(cfg, args, model, ckpt, logger, (label_pool, unlabel_pool),
                                                         100, rules=args.rules, score_plus=False, score_reverse=False)

