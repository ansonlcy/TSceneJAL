import glob
import os
import tqdm
import numpy as np
import torch
import random
import torch.nn as nn
from pcdet.datasets import build_al_dataloader
from pcdet.models import load_data_to_gpu, build_network, model_fn_decorator
import gc
from sklearn.cluster import kmeans_plusplus
import time
import faiss


def badge_strategies(cfg, args, model, ckpt_dir, logger, slice_set, search_num_each, au_theta=1):
    (label_pool, unlabel_pool) = slice_set

    _, _, unlabel_dataset, unlabel_dataloader = build_al_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        slice_set=slice_set,
        seed=666,
        workers=args.workers
    )

    lyft_flag = False
    if len(cfg.CLASS_NAMES) > 3:
        lyft_flag = True


    logger.info('**********************Start Badge search:label pool:%s  unlabel_pool:%s**********************' %
                (len(label_pool), len(unlabel_pool)))

    model.load_params_from_file(filename=ckpt_dir, logger=logger)
    model.cuda()
    model.eval()

    progress_bar = tqdm.tqdm(total=len(unlabel_dataloader), leave=True, desc='produce cls labels', dynamic_ncols=True)

    rpn_cls_results = {}
    for i, batch_dict in enumerate(unlabel_dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
            for idx, frame_id in enumerate(batch_dict['frame_id']):
                rpn_cls_preds = pred_dicts[idx]['rpn_cls_preds']
                rpn_cls_preds = torch.argmax(rpn_cls_preds.view(-1, model.dense_head.num_class), -1)
                rpn_cls_results[frame_id] = rpn_cls_preds.half().cpu()
        progress_bar.update()
    progress_bar.close()


    # next to get grads on models
    torch.cuda.empty_cache()
    model.train()
    _, _, grad_dataset, grad_dataloader = build_al_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        slice_set=slice_set,
        seed=666,
        workers=args.workers
    )
    progress_bar = tqdm.tqdm(total=len(grad_dataloader), leave=True, desc='get grad of cls', dynamic_ncols=True)

    rpn_cls_grad_embedding_list = []
    frame_id_list = []
    for i, batch_dict in enumerate(grad_dataloader):
        load_data_to_gpu(batch_dict)

        pred_dicts, _, _ = model(batch_dict)
        frame_id = batch_dict['frame_id'][0]
        cls_labels = rpn_cls_results[frame_id].unsqueeze(0).cuda()
        cls_preds = pred_dicts['rpn_cls_preds']

        new_data = {
            "box_cls_labels": cls_labels,
            "cls_preds": cls_preds
        }
        cls_loss = model.dense_head.get_cls_layer_loss(new_data=new_data)[0]
        loss = cls_loss
        model.zero_grad()
        loss.backward()

        # if has dropout layer
        if isinstance(model.dense_head.conv_cls, nn.Sequential):
            cls_grads = model.dense_head.conv_cls[-1].weight.grad.clone().detach().cpu()
        else:
            cls_grads = model.dense_head.conv_cls.weight.grad.clone().detach().cpu()

        # if use lyft dataset, the grad is so big, so we need to use half to save memory
        if lyft_flag:
            cls_grads = cls_grads.half()

        rpn_cls_grad_embedding_list.append(cls_grads)
        frame_id_list.append(frame_id)

        torch.cuda.empty_cache()

        progress_bar.update()
    progress_bar.close()
    model.zero_grad()
    model.eval()


    rpn_grad_embeddings = torch.stack(rpn_cls_grad_embedding_list, 0)
    del rpn_cls_grad_embedding_list
    gc.collect()
    num_sample = rpn_grad_embeddings.shape[0]
    rpn_grad_embeddings = rpn_grad_embeddings.view(num_sample, -1).numpy()
    start_time = time.time()

    # the feature and nums is so big in lyft, we use faiss to speed up
    if lyft_flag is True:
        dims = rpn_grad_embeddings.shape[1]
        kmeans = faiss.Kmeans(dims, search_num_each, niter=300, verbose=True, gpu=True)
        kmeans.train(rpn_grad_embeddings)

        index = faiss.IndexFlatL2(dims)
        index.add(rpn_grad_embeddings)

        _, selected_rpn_idx = index.search(kmeans.centroids, 1)
        selected_rpn_idx = selected_rpn_idx.reshape(-1)

    else:
        _, selected_rpn_idx = kmeans_plusplus(rpn_grad_embeddings, n_clusters=search_num_each, random_state=0)

    selected_frame_id = [frame_id_list[idx] for idx in selected_rpn_idx]
    logger.info("kmeans++ running time: %s seconds" % (time.time() - start_time))

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

