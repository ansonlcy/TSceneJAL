import glob
import os
import tqdm
import numpy as np
import torch
import random
import os
from pcdet.datasets import build_al_dataloader
from pcdet.models import load_data_to_gpu, build_network, model_fn_decorator
from pathlib import Path
import gc
from sklearn.cluster import kmeans_plusplus
import time
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn


def coreset_strategies(cfg, args, model, ckpt_dir, logger, slice_set, search_num_each, feature_save_dir=None):
    if feature_save_dir is None:
        feature_save_dir = Path(os.path.dirname(__file__)) / "feature"
        feature_save_dir.mkdir(parents=True, exist_ok=True)

    (label_pool, unlabel_pool) = slice_set

    logger.info('**********************Start Coreset search:label pool:%s  unlabel_pool:%s**********************' %
                (len(label_pool), len(unlabel_pool)))

    model.load_params_from_file(filename=ckpt_dir, logger=logger)
    model.cuda()
    model.eval()

    # if not kitti, the feature is so big, so we use maxpool3d
    lyft_flag = False
    if len(cfg.CLASS_NAMES) > 3:
        lyft_flag = True

    maxpool_2d = nn.MaxPool2d(kernel_size=(4, 4))
    maxpool_1d = nn.MaxPool1d(kernel_size=4)

    # get label data feature
    _, _, label_dataset, label_dataloader = build_al_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        slice_set=([], label_pool),
        seed=666,
        workers=args.workers
    )

    progress_bar = tqdm.tqdm(total=len(label_dataloader), leave=True, desc='get label data feature', dynamic_ncols=True)

    for i, batch_dict in enumerate(label_dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            model(batch_dict)
            for idx, frame_id in enumerate(batch_dict['frame_id']):
                _feature = batch_dict["spatial_features_2d"][idx].half()
                _feature = maxpool_2d(_feature)
                if lyft_flag:
                    _feature = maxpool_1d(_feature.permute(1, 2, 0)).permute(2, 0, 1)

                torch.save(_feature, feature_save_dir / f"{frame_id}.pt")
        progress_bar.update()
    progress_bar.close()

    # get unlabel data feature
    _, _, unlabel_dataset, unlabel_dataloader = build_al_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        slice_set=([], unlabel_pool),
        seed=666,
        workers=args.workers
    )

    progress_bar = tqdm.tqdm(total=len(unlabel_dataloader), leave=True, desc='get unlabel data feature',
                             dynamic_ncols=True)

    for i, batch_dict in enumerate(unlabel_dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            model(batch_dict)
            for idx, frame_id in enumerate(batch_dict['frame_id']):
                _feature = batch_dict["spatial_features_2d"][idx].half()
                _feature = maxpool_2d(_feature)
                if lyft_flag:
                    _feature = maxpool_1d(_feature.permute(1, 2, 0)).permute(2, 0, 1)
                torch.save(_feature, feature_save_dir / f"{frame_id}.pt")
        progress_bar.update()
    progress_bar.close()

    idx_to_id = {i: label_dataset.kitti_infos[i]['image']['image_idx'] for i in
                 range(len(label_dataset.kitti_infos))}
    label_id_list = [idx_to_id[i] for i in label_pool]
    unlabel_id_list = [idx_to_id[i] for i in unlabel_pool]

    frame_id_list = label_id_list + unlabel_id_list
    # frame_id_list = [n.name.split(".")[0] for n in feature_save_dir.glob("*.pt")]

    feature_dict = {}
    for i in tqdm.tqdm(range(len(frame_id_list)), desc="create feature dict"):
        feature_dict[frame_id_list[i]] = torch.load(feature_save_dir / f"{frame_id_list[i]}.pt", map_location="cpu")

    def pairwise_squared_distances(unlabel_id_list, label_id_list):
        distance_matrix = torch.zeros(len(unlabel_id_list), len(label_id_list))
        for i in range(len(unlabel_id_list)):
            # feature_i = torch.load(feature_save_dir / f"{unlabel_id_list[i]}.pt", map_location="cuda:0")
            feature_i = feature_dict[unlabel_id_list[i]].to(device="cuda:0")
            for j in range(len(label_id_list)):
                # feature_j = torch.load(feature_save_dir / f"{label_id_list[j]}.pt", map_location="cuda:0")
                feature_j = feature_dict[label_id_list[j]].to(device="cuda:0")
                distance_matrix[i, j] = torch.dist(feature_i, feature_j)

        # nan 设置为0
        distance_matrix[distance_matrix != distance_matrix] = 0

        return torch.clamp(distance_matrix, 0.0, float('inf'))

    def furthest_first(unlabel_id_list, label_id_list, sample_nums, logger):
        unlabel_nums = len(unlabel_id_list)

        logger.info("start furthest_first init dist_ctr")
        dist_ctr = pairwise_squared_distances(unlabel_id_list, label_id_list)
        logger.info("end furthest_first init dist_ctr")
        # min_dist: [num_unlabel]-D -> record the nearest distance to the orig X_set
        min_dist = dist_ctr.mean(1)
        idxs = []
        for i in tqdm.tqdm(range(sample_nums), desc="sampling frames"):
            # choose the furthest to orig X_set
            idx = torch.argmax(min_dist)
            idxs.append(idx)
            # no need to calculate if the last iter
            if i < (sample_nums - 1):
                dist_new_ctr = pairwise_squared_distances(unlabel_id_list, [unlabel_id_list[idx]])
                for j in range(unlabel_nums):
                    min_dist[j] = min(min_dist[j], dist_new_ctr[j])
        return idxs

    logger.info("start coreset")
    start_time = time.time()
    choose_idx = furthest_first(unlabel_id_list, label_id_list, search_num_each, logger)
    logger.info("Coreset time cost: %s" % (time.time() - start_time))

    selected_frame_id = [unlabel_id_list[i] for i in choose_idx]

    del feature_dict
    gc.collect()

    # create the dict map sample id to index in dataset pool
    id_to_idx = {unlabel_dataset.kitti_infos[i]['image']['image_idx']: i for i in
                 range(len(unlabel_dataset.kitti_infos))}

    choose_idx = [id_to_idx[i] for i in selected_frame_id]

    label_pool = list(set(label_pool) | (set(choose_idx)))
    unlabel_pool = list(set(unlabel_pool) - set(choose_idx))
    random.shuffle(label_pool)
    random.shuffle(unlabel_pool)

    logger.info('\n**********************End search:label pool:%s  unlabel_pool:%s**********************' %
                (len(label_pool), len(unlabel_pool)))

    return label_pool, unlabel_pool, selected_frame_id
