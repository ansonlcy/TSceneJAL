import glob
import os
import tqdm
import numpy as np
import torch
import random

from pcdet.datasets import build_al_dataloader
from pcdet.models import load_data_to_gpu, build_network, model_fn_decorator
from torch.distributions import Categorical
from scipy.stats import uniform
import gc
from sklearn.cluster import kmeans_plusplus
import time
from typing import Dict, List
import scipy
from sklearn.neighbors import KernelDensity


def crb_strategies(cfg, args, model, ckpt_dir, logger, slice_set, search_num_each, k1, k2, bandwidth=5):
    (label_pool, unlabel_pool) = slice_set

    _, _, unlabel_dataset, unlabel_dataloader = build_al_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        slice_set=slice_set,
        seed=666,
        workers=args.workers
    )

    id_to_idx = {unlabel_dataset.kitti_infos[i]['image']['image_idx']: i for i in
                 range(len(unlabel_dataset.kitti_infos))}
    idx_to_id = {value: key for key, value in id_to_idx.items()}

    logger.info('**********************Start crb search:label pool:%s  unlabel_pool:%s**********************' %
                (len(label_pool), len(unlabel_pool)))

    model.load_params_from_file(filename=ckpt_dir, logger=logger)
    model.cuda()
    model.eval()

    progress_bar = tqdm.tqdm(total=len(unlabel_dataloader), leave=True, desc='produce cls and box labels',
                             dynamic_ncols=True)

    gt_labels = {}
    select_dic = {}
    density_list = {}
    label_list = {}

    '''
    -------------  Stage 1: Consise Label Sampling ----------------------
    '''

    for i, batch_dict in enumerate(unlabel_dataloader):
        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            preds_dict, _ = model(batch_dict)

            # use mc dropout to get the predict result (as pseudo labels)
            # this is for Stage 2
            mc_preds_dict, _ = model.post_processing(batch_dict, mc_preds=True)

            # cal points density this for stage 3
            points_preds_dict, _ = model.post_processing(batch_dict, cal_density=True)

            for idx, frame_id in enumerate(batch_dict['frame_id']):

                # get the entropy of frames, this is for stage 1
                value, counts = torch.unique(preds_dict[idx]['pred_labels'], return_counts=True)
                if len(value) == 0:
                    entropy = 0
                else:
                    # calculates the shannon entropy of the predicted labels of bounding boxes
                    unique_proportions = torch.ones(model.dense_head.num_class).cuda()
                    unique_proportions[value - 1] = counts.float()
                    entropy = Categorical(probs=unique_proportions / sum(counts)).entropy()
                select_dic[frame_id] = entropy

                # use mc dropout to get the predict result (as pseudo labels)
                # this is for Stage 2
                mc_box_preds = torch.cat(
                    (mc_preds_dict[idx]['pred_boxes'], mc_preds_dict[idx]['pred_labels'].unsqueeze(-1)), dim=-1)
                mc_box_mask = mc_preds_dict[idx]['pred_scores'] >= 0.3
                mc_box_preds = mc_box_preds[mc_box_mask]
                gt_labels[frame_id] = mc_box_preds.cpu()

                # cal the points density
                density_list[frame_id] = points_preds_dict[idx]['pred_box_points_density']
                label_list[frame_id] = points_preds_dict[idx]['pred_labels']
        progress_bar.update()
    progress_bar.close()

    # sort and get selected_frames
    select_dic = dict(sorted(select_dic.items(), key=lambda item: item[1]))
    # narrow down the scope
    selected_frames_s1 = list(select_dic.keys())[::-1][:int(k1 * search_num_each)]
    selected_idx_s1 = [id_to_idx[frame_id] for frame_id in selected_frames_s1]

    '''
    -------------  Stage 2: Representative Prototype Selection ----------------------
    '''

    # next to get grads on models
    torch.cuda.empty_cache()
    model.train()
    _, _, grad_dataset, grad_dataloader = build_al_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        slice_set=([], selected_idx_s1),
        seed=666,
        workers=args.workers
    )
    progress_bar = tqdm.tqdm(total=len(grad_dataloader), leave=True, desc='get grad of backone', dynamic_ncols=True)

    grad_embedding_list = []
    frame_id_list_s2 = []
    for i, batch_dict in enumerate(grad_dataloader):
        model.zero_grad()
        load_data_to_gpu(batch_dict)

        frame_id = batch_dict['frame_id'][0]
        gt_boxes = gt_labels[frame_id].cuda()
        # change the gt labels in batch_dict
        batch_dict['gt_boxes'] = gt_boxes.unsqueeze(0)

        ret_dict, _, _ = model(batch_dict)
        loss = ret_dict['loss'].mean()
        loss.backward()

        backbone_grad = model.backbone_2d.blocks[-1][-3].weight.grad.clone()

        # if nan, set 0
        backbone_grad[backbone_grad != backbone_grad] = 0
        backbone_grad = backbone_grad.view(-1).detach().cpu()

        grad_embedding_list.append(backbone_grad)
        frame_id_list_s2.append(frame_id)

        torch.cuda.empty_cache()

        progress_bar.update()
    progress_bar.close()
    model.zero_grad()
    model.eval()

    fc_grad_embeddings = torch.stack(grad_embedding_list, 0)

    del grad_embedding_list
    gc.collect()

    start_time = time.time()

    # choose the prefered prototype selection method and select the K2 medoids

    _, selected_fc_idx = kmeans_plusplus(fc_grad_embeddings.numpy(),
                                         n_clusters=int(k2 * search_num_each),
                                         random_state=0)

    selected_frames_s2 = [frame_id_list_s2[i] for i in selected_fc_idx]

    logger.info("running time: %s seconds for grads sampling " % (time.time() - start_time))

    '''
    -------------  Stage 3: Greedy Point Cloud Density Balancing ----------------------
    '''

    sampled_density_list = [density_list[i] for i in selected_frames_s2]
    sampled_label_list = [label_list[i] for i in selected_frames_s2]

    """ Build the uniform distribution for each class """
    start_time = time.time()
    density_all = torch.cat(list(density_list.values()), 0)
    label_all = torch.cat(list(label_list.values()), 0)
    unique_labels, label_counts = torch.unique(label_all, return_counts=True)
    sorted_density = [torch.sort(density_all[label_all == unique_label])[0] for unique_label in unique_labels]
    # global_density_max eg.[288, 1971, 913] 三个类别的在全局中（待选的数据中）最大密度
    global_density_max = [int(sorted_density[unique_label][-1]) for unique_label in range(len(unique_labels))]
    # high 为每一类在95%位置（总数量95%的地方）的密度，low为每一类在5%位置的密度
    alpha = 0.95
    global_density_high = [int(sorted_density[unique_label][int(alpha * label_counts[unique_label])]) for
                           unique_label in range(len(unique_labels))]
    global_density_low = [int(sorted_density[unique_label][-int(alpha * label_counts[unique_label])]) for
                          unique_label in range(len(unique_labels))]
    x_axis = [np.linspace(-50, int(global_density_max[i]) + 50, 400) for i in range(model.dense_head.num_class)]
    uniform_dist_per_cls = [
        uniform.pdf(x_axis[i], global_density_low[i], global_density_high[i] - global_density_low[i]) for i in
        range(model.dense_head.num_class)]

    print("--- Build the uniform distribution running time: %s seconds ---" % (time.time() - start_time))

    density_list, label_list, frame_id_list_s3 = sampled_density_list, sampled_label_list, selected_frames_s2
    selected_frames_s3: List[str] = []
    selected_box_densities: torch.tensor = torch.tensor([]).cuda()
    selected_box_labels: torch.tensor = torch.tensor([]).cuda()
    # looping over N_r samples

    pbar = tqdm.tqdm(total=search_num_each, leave=True, desc='points density search', dynamic_ncols=True)

    for j in range(search_num_each):
        if j == 0:  # initially, we randomly select a frame.

            selected_frames_s3.append(frame_id_list_s3[j])
            selected_box_densities = torch.cat((selected_box_densities, density_list[j]))
            selected_box_labels = torch.cat((selected_box_labels, label_list[j]))

            # remove selected frame
            del density_list[0]
            del label_list[0]
            del frame_id_list_s3[0]

        else:  # go through all the samples and choose the frame that can most reduce the KL divergence
            best_frame_id = None
            best_frame_index = None
            best_inverse_coff = -1

            for i in range(len(density_list)):
                unique_proportions = np.zeros(model.dense_head.num_class)
                KL_scores_per_cls = np.zeros(model.dense_head.num_class)
                for cls in range(model.dense_head.num_class):
                    if (label_list[i] == cls + 1).sum() == 0:
                        unique_proportions[cls] = 1
                        KL_scores_per_cls[cls] = np.inf
                    else:
                        # get existing selected box densities
                        selected_box_densities_cls = selected_box_densities[selected_box_labels == (cls + 1)]
                        # append new frame's box densities to existing one
                        selected_box_densities_cls = torch.cat((selected_box_densities_cls,
                                                                density_list[i][label_list[i] == (cls + 1)]))
                        # initialize kde
                        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
                            selected_box_densities_cls.cpu().numpy()[:, None])

                        logprob = kde.score_samples(x_axis[cls][:, None])
                        KL_score_per_cls = scipy.stats.entropy(uniform_dist_per_cls[cls], np.exp(logprob))
                        KL_scores_per_cls[cls] = KL_score_per_cls
                        # ranging from 0 to 1
                        unique_proportions[cls] = 2 / np.pi * np.arctan(np.pi / 2 * KL_score_per_cls)

                inverse_coff = np.mean(1 - unique_proportions)
                # KL_save_list.append(inverse_coff)
                if inverse_coff > best_inverse_coff:
                    best_inverse_coff = inverse_coff
                    best_frame_index = i
                    best_frame_id = frame_id_list_s3[i]

            # remove selected frame
            selected_box_densities = torch.cat((selected_box_densities, density_list[best_frame_index]))
            selected_box_labels = torch.cat((selected_box_labels, label_list[best_frame_index]))
            del density_list[best_frame_index]
            del label_list[best_frame_index]
            del frame_id_list_s3[best_frame_index]

            selected_frames_s3.append(best_frame_id)

        pbar.update()
        pbar.refresh()
    pbar.close()

    choose_idx = [id_to_idx[i] for i in selected_frames_s3]
    label_pool = list(set(label_pool) | (set(choose_idx)))
    unlabel_pool = list(set(unlabel_pool) - set(choose_idx))
    random.shuffle(label_pool)
    random.shuffle(unlabel_pool)

    logger.info('\n**********************End search:label pool:%s  unlabel_pool:%s**********************' %
                    (len(label_pool), len(unlabel_pool)))


    return label_pool, unlabel_pool, selected_frames_s3
