import glob
import os
import tqdm
import numpy as np
import torch
import random

from pcdet.datasets import build_al_dataloader
from pcdet.models import load_data_to_gpu, build_network, model_fn_decorator

from pcdet.utils.al_utils import calculate_category_entropy


def mdn_strategies(cfg, args, model, ckpt_dir, logger, slice_set, search_num_each, rules=1, score_thresh=0.3,
                   eu_theta=1,
                   au_theta=1, score_plus=False, score_reverse=False, consider_other=True, k1=3):
    """

    :param cfg:
    :param args:
    :param model:
    :param ckpt_dir:
    :param logger:
    :param slice_set:
    :param search_num_each: 每一轮主动学习搜索的数量
    :param rules: int, 选择策略
    :param score_thresh: 对预测结果根据阈值过滤，一些低置信度的结果会对最后的选择产生影响，默认是0.3
    :param au_theta: 当使用au+eu策略时，au的权重
    :param score_plus: 在使用au策略时，是否使用置信度加权，即 score*au
    :param score_reverse: 在使用au策略时，是否使用1-置信度加权，即 (1-score)*au
    :param consider_other: 选择的时候是否考虑其他类，默认不考虑
    :param k1: 如果使用两阶段策略（先根据类别熵进行第一阶段选择），第一阶段的选择数量相较于search_num_each的比例
    :return:
    """
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

    au_list = []
    eu_list = []
    score_list = []
    id_list = []
    car_num_list = []
    cate_entropy_list = []  # the category entropy
    empty_list = []  # the empty frame

    for idx, det in enumerate(det_annos):
        if len(det['name'].tolist()) == 0:
            empty_list.append(det['frame_id'])
        else:
            choose_idx = det['score'] >= score_thresh
            if np.sum(choose_idx) == 0:  # means threre is no car (with score >= thresh) in this frame
                empty_list.append(det['frame_id'])
                continue

            # cal the category entropy
            cate_entropy = calculate_category_entropy(det['name'][choose_idx])
            cate_entropy_list.append(cate_entropy)

            if det['al'].ndim == 1:  # if we use corner loss varance, the shape of al is (N,)
                au_list.append(det['al'][choose_idx])
            else:
                au_list.append(np.max(det['al'][choose_idx], axis=1))

            score_list.append(det['score'][choose_idx])
            car_num_list.append(len(choose_idx))

            if rules in [6, 7, 9]:  # use eu or eu + au
                # if we use mdn than the eu is calulated by each box, so it has the same length of au, and the type is ndarray
                if isinstance(det['ep'], np.ndarray):
                    eu_list.append(det['ep'][choose_idx])
                # if use mc dropout, we only get one value for each frame
                else:
                    eu_list.append(float(det['ep']))

            id_list.append(det['frame_id'])

    id_list = np.array(id_list)

    # use 0-1 normalize on au data
    max_al_uc = np.max([val for sub in au_list for val in sub])
    min_al_uc = np.min([val for sub in au_list for val in sub])
    # some trick
    if score_reverse:
        for i in range(len(score_list)):
            score_list[i] = 1 - score_list[i]
    if score_plus:
        for i in range(len(au_list)):
            # au_list[i] = ((au_list[i] - mean_al_uc) / stdev_al_uc) * score_list[i]
            au_list[i] = ((au_list[i] - min_al_uc) / (max_al_uc - min_al_uc)) * score_list[i]
    else:
        for i in range(len(au_list)):
            # au_list[i] = (au_list[i] - mean_al_uc) / stdev_al_uc
            au_list[i] = (au_list[i] - min_al_uc) / (max_al_uc - min_al_uc)

    if rules == 11:
        au_list = score_list

    au_list = combine_uncer_in_frame(au_list)

    # use 0-1 on eu data if we use eu or eu + au
    if len(eu_list) > 0 and isinstance(eu_list[0], float):
        eu_list = np.array(eu_list)
        eu_list = (eu_list - np.min(eu_list)) / (np.max(eu_list) - np.min(eu_list))
        # we need to use 0-1 normalize again on au_list because we use eu + au
        au_list = (au_list - np.min(au_list)) / (np.max(au_list) - np.min(au_list))

    if len(eu_list) > 0 and isinstance(eu_list[0], np.ndarray):
        max_ep_uc = np.max([val for sub in eu_list for val in sub])
        min_ep_uc = np.min([val for sub in eu_list for val in sub])
        for i in range(len(eu_list)):
            eu_list[i] = ((eu_list[i] - min_ep_uc) / (max_ep_uc - min_ep_uc))
        eu_list = combine_uncer_in_frame(eu_list, rules=rules)
        # use 0-1 again on eu and au
        au_list = (au_list - np.min(au_list)) / (np.max(au_list) - np.min(au_list))
        eu_list = (eu_list - np.min(eu_list)) / (np.max(eu_list) - np.min(eu_list))

    # next to choose the score_list (au or eu or eu + au)
    if rules in [7, 9]:
        # choose_score_list = eu_list + eu_theta * au_list
        choose_score_list = eu_theta * eu_list + au_theta * au_list
    elif rules == 6:
        choose_score_list = eu_list
    else:
        choose_score_list = au_list

    # 如果使用二阶段策略
    if rules == 9:
        cate_entropy_list = np.array(cate_entropy_list)
        sorted_indices_k1 = np.argsort(cate_entropy_list)[::-1][:int(k1 * search_num_each)]

        choose_score_list = choose_score_list[sorted_indices_k1]
        id_list = id_list[sorted_indices_k1]

    # 直接使用类别熵作为筛选依据
    if rules == 10:
        choose_score_list = np.array(cate_entropy_list)


    sorted_indices = np.argsort(choose_score_list)[::-1]
    # this sort list not include the empty frame
    sort_id_list = id_list[sorted_indices]
    # next we need to add the empty frame
    # sort_id_list = np.concatenate((sort_id_list, np.array(empty_list)), axis=0)

    choose_id = sort_id_list[:search_num_each]

    # create the dict map sample id to index in dataset pool
    id_to_idx = {unlabel_dataset.kitti_infos[i]['image']['image_idx']: i for i in
                 range(len(unlabel_dataset.kitti_infos))}

    choose_idx = [id_to_idx[i] for i in choose_id]

    label_pool = list(set(label_pool) | (set(choose_idx)))
    unlabel_pool = list(set(unlabel_pool) - set(choose_idx))
    random.shuffle(label_pool)
    random.shuffle(unlabel_pool)

    if hasattr(unlabel_dataset, 'use_shared_memory') and unlabel_dataset.use_shared_memory:
        unlabel_dataset.clean_shared_memory()

    logger.info('\n**********************End search:label pool:%s  unlabel_pool:%s**********************' %
                (len(label_pool), len(unlabel_pool)))

    return label_pool, unlabel_pool, choose_id


def combine_uncer_in_frame(uncertain_list):
    """
    :param uncertain_list: list [ndarray]
    :return:
    """
    for i in range(len(uncertain_list)):
        uncertain_list[i] = np.mean(uncertain_list[i])

    return np.array(uncertain_list)
