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