import glob
import os
import tqdm
import numpy as np
import torch
import random

from pcdet.datasets import build_al_dataloader
from pcdet.models import load_data_to_gpu, build_network, model_fn_decorator

from pcdet.utils.al_utils import calculate_category_entropy
from pcdet.query_strategies.graph_utils import filter_process_det_by_id, get_graph_from_frame, similarity_among_graphs, farthest_point_sampling

os.environ['PATH'] = os.environ['PATH'] + ':/usr/local/cuda/bin'


def similarity_strategies(cfg, args, model, ckpt_dir, logger, slice_set, search_num_each, rules=1, score_thresh=0.3, eu_theta=1,
                          au_theta=1, k1=3, k2=2):
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
    :param eu_theta: 当使用au+eu策略时，eu的权重
    :param au_theta: 当使用au+eu策略时，au的权重
    :param consider_other: 选择的时候是否考虑其他类，默认不考虑
    :param k1: 如果使用两阶段策略（先根据类别熵进行第一阶段选择），第一阶段的选择数量相较于search_num_each的比例
    :param k2:
    :return:
    """
    (label_pool, unlabel_pool) = slice_set

    dataset_name = "kitti"
    lyft_flag = False
    if len(cfg.CLASS_NAMES) > 3:
        lyft_flag = True
        dataset_name = 'lyft'


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

    # the type we consider

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


            eu_list.append(det['ep'][choose_idx])
            id_list.append(det['frame_id'])

    id_list = np.array(id_list)

    # use 0-1 normalize on au data
    max_al_uc = np.max([val for sub in au_list for val in sub])
    min_al_uc = np.min([val for sub in au_list for val in sub])
    for i in range(len(au_list)):
        au_list[i] = (au_list[i] - min_al_uc) / (max_al_uc - min_al_uc)
    au_list = combine_uncer_in_frame(au_list, rules=rules)
    au_list = (au_list - np.min(au_list)) / (np.max(au_list) - np.min(au_list))

    # use 0-1 normalize on eu data
    max_ep_uc = np.max([val for sub in eu_list for val in sub])
    min_ep_uc = np.min([val for sub in eu_list for val in sub])
    for i in range(len(eu_list)):
        eu_list[i] = ((eu_list[i] - min_ep_uc) / (max_ep_uc - min_ep_uc))
    eu_list = combine_uncer_in_frame(eu_list, rules=rules)
    eu_list = (eu_list - np.min(eu_list)) / (np.max(eu_list) - np.min(eu_list))

    # next to choose the score_list (au or eu or eu + au)
    choose_score_list = eu_theta * eu_list + au_theta * au_list


    # 如果使用三阶段策略
    if rules == 13:

        # 1. cate entropy sampling
        cate_entropy_list = np.array(cate_entropy_list)
        sorted_indices_k1 = np.argsort(cate_entropy_list)[::-1][:int(k1 * search_num_each)]
        choose_score_list = choose_score_list[sorted_indices_k1]
        id_list = id_list[sorted_indices_k1]
        logger.info(f"**********************3 strageys:cate sampling {len(id_list)}**********************")


        # 2. uncertainty sampling
        sorted_indices_k2 = np.argsort(choose_score_list)[::-1][:int(k2 * search_num_each)]
        choose_score_list = choose_score_list[sorted_indices_k2]
        id_list = id_list[sorted_indices_k2]
        logger.info(f"**********************3 strageys:uncer sampling {len(id_list)}**********************")


        # 3. farthest sampling

        process_det = filter_process_det_by_id(dataset_name, det_annos, id_list, score_thresh=0.3)
        graph_list = []
        for key in id_list:
            graph_list.append(get_graph_from_frame(process_det[key]))
        distance_metrix = 1 - similarity_among_graphs(graph_list)
        choose_id, _ = farthest_point_sampling(id_list, distance_metrix, search_num_each)
        logger.info(f"**********************3 strageys:farthest sampling {len(choose_id)}**********************")

    # 如果使用三阶段，但是第二阶段使用smilarity sampling
    elif rules == 14:
        # 1. cate entropy sampling
        cate_entropy_list = np.array(cate_entropy_list)
        sorted_indices_k1 = np.argsort(cate_entropy_list)[::-1][:int(k1 * search_num_each)]
        choose_score_list = choose_score_list[sorted_indices_k1]
        id_list = id_list[sorted_indices_k1]
        logger.info(f"**********************3 strageys:cate sampling {len(id_list)}**********************")

        # 2. farthest sampling
        process_det = filter_process_det_by_id(dataset_name, det_annos, id_list, score_thresh=0.3)
        graph_list = []
        for key in id_list:
            graph_list.append(get_graph_from_frame(process_det[key]))
        distance_metrix = 1 - similarity_among_graphs(graph_list)
        choose_id, _ = farthest_point_sampling(id_list, distance_metrix, int(k2 * search_num_each))

        score_idx = [np.where(id_list == e)[0][0] for e in choose_id]

        choose_score_list = choose_score_list[score_idx]
        logger.info(f"**********************3 strageys:farthest sampling {len(choose_id)}**********************")

        # 3. uncertainty sampling
        sorted_indices_k3 = np.argsort(choose_score_list)[::-1][:int(search_num_each)]
        choose_score_list = choose_score_list[sorted_indices_k3]
        choose_id = np.array(choose_id)[sorted_indices_k3]
        logger.info(f"**********************3 strageys:uncer sampling {len(choose_id)}**********************")


    # 如果使用二阶段，第二阶段使用smilarity
    elif rules == 15:
        # 1. cate entropy sampling
        cate_entropy_list = np.array(cate_entropy_list)
        sorted_indices_k1 = np.argsort(cate_entropy_list)[::-1][:int(k1 * search_num_each)]
        choose_score_list = choose_score_list[sorted_indices_k1]
        id_list = id_list[sorted_indices_k1]
        logger.info(f"**********************2 strageys:cate sampling {len(id_list)}**********************")

        # 2. farthest sampling
        process_det = filter_process_det_by_id(dataset_name, det_annos, id_list, score_thresh=0.3)
        graph_list = []
        for key in id_list:
            graph_list.append(get_graph_from_frame(process_det[key]))
        distance_metrix = 1 - similarity_among_graphs(graph_list)
        choose_id, _ = farthest_point_sampling(id_list, distance_metrix, search_num_each)
        logger.info(f"**********************2 strageys:similarity sampling {len(choose_id)}**************")

    # 如果使用二阶段，第二阶段使用random
    elif rules == 152:
        # 1. cate entropy sampling
        cate_entropy_list = np.array(cate_entropy_list)
        sorted_indices_k1 = np.argsort(cate_entropy_list)[::-1][:int(k1 * search_num_each)]
        choose_score_list = choose_score_list[sorted_indices_k1]
        id_list = id_list[sorted_indices_k1]
        logger.info(f"**********************2 strageys:cate sampling {len(id_list)}**********************")

        # 2. random_choose
        choose_id = random.sample(list(id_list), int(search_num_each))
        logger.info(f"**********************2 strageys:random sampling {len(choose_id)}**************")

    # 如果使用二阶段，第二阶段使用fs 但是不使用固定中心点方法
    elif rules == 153:
        # 1. cate entropy sampling
        cate_entropy_list = np.array(cate_entropy_list)
        sorted_indices_k1 = np.argsort(cate_entropy_list)[::-1][:int(k1 * search_num_each)]
        choose_score_list = choose_score_list[sorted_indices_k1]
        id_list = id_list[sorted_indices_k1]
        logger.info(f"**********************2 strageys:cate sampling {len(id_list)}**********************")

        # 2. fasthest sampling with out fix init point
        process_det = filter_process_det_by_id(dataset_name, det_annos, id_list, score_thresh=0.3)
        graph_list = []
        for key in id_list:
            graph_list.append(get_graph_from_frame(process_det[key]))
        distance_metrix = 1 - similarity_among_graphs(graph_list)
        choose_id, _ = farthest_point_sampling(id_list, distance_metrix, search_num_each, init_sample=False)
        logger.info(f"**********************2 strageys:similarity sampling(w_o init sample) {len(choose_id)}**************")


    # 如果使用三阶段，第一阶段使用theta*au+eu，第二阶段使用similarity, 第三阶段使用类别熵
    elif rules == 16:
        # 1. uncertainy sampling
        uncertainy_list = np.array(choose_score_list)
        cate_entropy_list = np.array(cate_entropy_list)
        sorted_indices_k1 = np.argsort(uncertainy_list)[::-1][:int(k1 * search_num_each)]
        choose_entropy_list = cate_entropy_list[sorted_indices_k1]
        id_list = id_list[sorted_indices_k1]
        logger.info(f"**********************3 strageys:uncertainty sampling {len(id_list)}**********************")

        # 2. farthest sampling
        process_det = filter_process_det_by_id(dataset_name, det_annos, id_list, score_thresh=0.3)
        graph_list = []
        for key in id_list:
            graph_list.append(get_graph_from_frame(process_det[key]))
        distance_metrix = 1 - similarity_among_graphs(graph_list)
        choose_id, _ = farthest_point_sampling(id_list, distance_metrix, int(k2 * search_num_each))

        score_idx = [np.where(id_list == e)[0][0] for e in choose_id]

        choose_entropy_list = choose_entropy_list[score_idx]
        logger.info(f"**********************3 strageys:farthest sampling {len(choose_id)}**********************")

        # 3. cate entropy sampling
        sorted_indices_k3 = np.argsort(choose_entropy_list)[::-1][:int(search_num_each)]
        choose_entropy_list = choose_entropy_list[sorted_indices_k3]
        choose_id = np.array(choose_id)[sorted_indices_k3]
        logger.info(f"**********************3 strageys:category entropy sampling {len(choose_id)}**********************")

    # 如果使用三阶段，第一阶段使用theta*au+eu，第二阶段使用类别熵, 第三阶段使用similarity
    elif rules == 17:
        # 1. uncertainy sampling
        uncertainy_list = np.array(choose_score_list)
        cate_entropy_list = np.array(cate_entropy_list)
        sorted_indices_k1 = np.argsort(uncertainy_list)[::-1][:int(k1 * search_num_each)]
        choose_entropy_list = cate_entropy_list[sorted_indices_k1]
        id_list = id_list[sorted_indices_k1]
        logger.info(f"**********************3 strageys:uncertainty sampling {len(id_list)}**********************")

        # 2. cate entropy sampling
        sorted_indices_k2 = np.argsort(choose_entropy_list)[::-1][:int(k2 * search_num_each)]
        choose_entropy_list = choose_entropy_list[sorted_indices_k2]
        choose_id = id_list[sorted_indices_k2]
        logger.info(f"**********************3 strageys:category entropy sampling {len(choose_id)}**********************")

        # 3. farthest sampling
        process_det = filter_process_det_by_id(dataset_name, det_annos, choose_id, score_thresh=0.3)
        graph_list = []
        for key in choose_id:
            graph_list.append(get_graph_from_frame(process_det[key]))
        distance_metrix = 1 - similarity_among_graphs(graph_list)
        choose_id, _ = farthest_point_sampling(choose_id, distance_metrix, int(search_num_each))
        logger.info(f"**********************3 strageys:farthest sampling {len(choose_id)}**********************")


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


def combine_uncer_in_frame(uncertain_list, rules):
    """

    :param uncertain_list: list [ndarray]
    :param rules:
    :return:
    """
    for i in range(len(uncertain_list)):
        if rules == 1:
            uncertain_list[i] = np.max(uncertain_list[i])
        elif rules == 2 or rules == 8:
            uncertain_list[i] = np.sum(uncertain_list[i])
        else:
            # if the rule is 6 or 7, we use eu or eu + au, so we need to use mean the express the au of frame
            uncertain_list[i] = np.mean(uncertain_list[i])

    return np.array(uncertain_list)
