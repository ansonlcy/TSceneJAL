from pcdet.datasets import build_al_dataloader
import random
import numpy as np


def random_change_data_pool(logger, slice_set, search_num_each):
    (label_pool, unlabel_pool) = slice_set
    random.shuffle(label_pool)
    random.shuffle(unlabel_pool)
    choose = unlabel_pool[:search_num_each]

    logger.info('**********************Start random search:label pool:%s  unlabel_pool:%s**********************' %
                (len(label_pool), len(unlabel_pool)))

    label_pool = list(set(label_pool) | (set(choose)))
    unlabel_pool = list(set(unlabel_pool) - set(choose))

    logger.info('\n**********************End search:label pool:%s  unlabel_pool:%s**********************' %
                (len(label_pool), len(unlabel_pool)))

    return label_pool, unlabel_pool


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