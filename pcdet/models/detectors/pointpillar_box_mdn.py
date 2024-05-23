import torch
from .detector3d_template import Detector3DTemplate
from ..model_utils import model_nms_utils
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils
from scipy.stats import norm
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class PointPillarBoxMdn(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }

            if batch_dict.get("rpn_cls_preds", None) is not None:
                ret_dict['rpn_cls_preds'] = batch_dict['rpn_cls_preds']
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict, mc_preds=False, cal_density=False):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                batch_box_preds_var: (B, num_boxes, 8, 3), 8 corners and each corner has 3 var(x,y,z)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)

            mc_preds: if use monte carlo sampling results to replace the original predictions
            cal_density: whether calculate the density of point cloud for each predicted box
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            if not mc_preds:
                box_preds = batch_dict['batch_box_preds'][batch_mask]
            else:
                box_preds = batch_dict['batch_box_preds_mc'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):

                if not mc_preds:
                    cls_preds = batch_dict['batch_cls_preds'][batch_mask]
                else:
                    cls_preds = batch_dict['batch_cls_preds_mc'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                if not mc_preds:
                    cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                else:
                    cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds_mc']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
                final_vars = batch_dict['batch_box_preds_var'][batch_mask][selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            # next is to process the mdn if we use it
            if 'batch_box_preds_pi' and 'batch_box_preds_raw' in batch_dict:
                final_boxes_raw = batch_dict['batch_box_preds_raw'][batch_mask][selected]
                final_pi = batch_dict['batch_box_preds_pi'][batch_mask][selected]
                # next to process the varance
                final_vars, var_au, var_eu = self.get_uncertainty_box_mdn(final_boxes, final_boxes_raw, final_pi, final_vars,
                                                                         box_ver_method='mean')

            recall_dict = self.generate_picp_record(box_preds=final_boxes, var_preds=final_vars,
                                                    data_dict=batch_dict,
                                                    recall_dict=recall_dict, batch_index=index, iou_thresh=0.5
                                                    , alpha=0.05)

            record_dict = {}
            if self.model_cfg.DENSE_HEAD.get('MC_SAMPLE_NUMS', None) is not None:
                # process the mc dropout result
                cls_preds_mc = batch_dict['batch_cls_preds_mc_var'][batch_mask]
                box_preds_mc = batch_dict['batch_box_preds_mc_var'][batch_mask]
                # if not batch_dict['cls_preds_normalized']:
                #     cls_preds_mc = torch.sigmoid(cls_preds_mc)
                #     cls_preds_mc = torch.max(cls_preds_mc, dim=-1)[0]
                # cls_mc_var = torch.var(cls_preds_mc, dim=0).mean()
                # box_mc_var = torch.var(box_preds_mc, dim=0).mean()
                # mc_var = cls_mc_var + box_mc_var
                # record_dict['mc_var'] = mc_var
                cls_mc_var = cls_preds_mc.mean()
                box_mc_var = box_preds_mc.mean()
                record_dict['mc_var'] = box_mc_var

            record_dict.update({
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
                'pred_vars': final_vars,


            })

            if batch_dict.get("rpn_cls_preds", None) is not None:
                record_dict.update(
                    {
                        'rpn_cls_preds': batch_dict["rpn_cls_preds"][batch_mask]
                    }
                )

            if 'batch_box_preds_pi' and 'batch_box_preds_raw' in batch_dict:
                record_dict.update({
                    'pred_pi': final_pi,
                    'box_au': var_au,
                    'box_eu': var_eu
                })

            if cal_density:

                src_points = batch_dict['points'][:, 1:4]
                batch_indices = batch_dict['points'][:, 0].long()
                bs_mask = (batch_indices == index)
                # get point clouds for each sample (to validate active label)
                sampled_points = src_points[bs_mask]
                sampled_points = sampled_points.reshape(1, sampled_points.shape[0], 3)


                pred_boxes_reshaped = final_boxes.reshape(1, -1, 7)
                box_volumes = final_boxes[:, 3] * final_boxes[:, 4] * final_boxes[:, 5]
                pred_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(sampled_points,
                                                                                 pred_boxes_reshaped).long().squeeze(
                    dim=0)
                pts_counts = torch.zeros(final_boxes.shape[0] + 1).cuda()
                unique_box_id, pred_box_unique_pts_counts = torch.unique(pred_box_idxs_of_pts,
                                                                         return_counts=True)  # remove background: -1
                for i, box_id in enumerate(unique_box_id):
                    pts_counts[box_id] = pred_box_unique_pts_counts[i]
                record_dict["pred_box_points_density"] = pts_counts[:-1].cuda() / box_volumes

            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

    @staticmethod
    def get_uncertainty_box_mdn(pred_box, pred_box_raw, pred_pi, pred_var, box_ver_method='mean'):
        """
        :param pred_box: (N, 7)
        :param pred_box_raw: (N, 7, NUM_GAUSS)
        :param pred_pi: (N, 7, NUM_GAUSS)
        :param pred_var: (N, 7, NUM_GAUSS)
        :param box_ver_method: how to generate a box varance from 7 (x,y,z,l,w,h,yaw)
        :return:
        """
        var_eu = (pred_pi * ((pred_box_raw - pred_box.unsqueeze(-1)) ** 2)).sum(dim=-1)
        var_au = (pred_pi * pred_var).sum(dim=-1)
        var_split = var_au + var_eu

        if box_ver_method == 'mean':
            var_eu = var_eu.mean(dim=-1)
            var_au = var_au.mean(dim=-1)
        elif box_ver_method == 'max':
            var_eu = var_eu.max(dim=-1)
            var_au = var_au.max(dim=-1)
        elif box_ver_method == 'sum':
            var_eu = var_eu.sum(dim=-1)
            var_au = var_au.sum(dim=-1)

        return var_split, var_au, var_eu

    @staticmethod
    def generate_picp_record(box_preds, var_preds, data_dict, recall_dict, batch_index, iou_thresh, alpha):
        """

        :param box_preds: tensor (N, 7)
        :param var_preds: tensor (N, 8, 3), 8 corners and each corner has 3 varance(1 varance, var_x=var_y=var_z)
                        or (N, 7), the varance of (x, y, z, w, l, h, theta)
        :param iou_thresh: the thresh when assign the gt to the pred(default 0.5)
        :param alpha: float, percentage, means the 1-alpha will in the range (default 5%=0.05)
        :return:
        """
        if 'gt_boxes' not in data_dict:
            return recall_dict

        gt_boxes = data_dict['gt_boxes'][batch_index]

        last_len = 24
        if var_preds.shape[-1] == 3:
            last_len = 8 * 3
        elif var_preds.shape[-1] == 7:
            last_len = 7

        if "picp_cnt" not in recall_dict:
            recall_dict.update({"picp_cnt": torch.zeros(last_len).cuda(),
                                "pred_cnt": 0,
                                "mpiw_sum": torch.zeros(last_len).cuda(),
                                "uncer_type": None})

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        attr_cnt = var_preds.shape[-1]
        uncer_type = {3: "corner", 7: "center"}[attr_cnt]

        if cur_gt.shape[0] > 0 and box_preds.shape[0] > 0:
            # NOTE assign gt box a pred box whose iou > thresh
            # (N,7) (M,7) ->  N x M

            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            max_iou, gt_index = torch.max(iou3d, dim=1)
            box_preds = box_preds[max_iou > iou_thresh]
            var_preds = var_preds[max_iou > iou_thresh]
            gt_index = gt_index[max_iou > iou_thresh]
            gt_boxes = gt_boxes[gt_index][:, 0:7]

            if gt_boxes.shape[0] > 0 and box_preds.shape[0] > 0:
                if uncer_type == "corner":
                    box_preds = box_utils.boxes_to_corners_3d(box_preds)
                    gt_boxes = box_utils.boxes_to_corners_3d(gt_boxes)
                elif uncer_type == "center":
                    pass
                else:
                    raise NotImplementedError

                # note that if attr_cnt=3, means 8 corners and each corner has 3 var, thus we need to view (-1, 8*3)
                mean = box_preds.view(-1, last_len)
                sigma = var_preds.sqrt().view(-1, last_len)
                gt_boxes = gt_boxes.view(-1, last_len)

                interval = sigma.mul(norm.ppf(alpha / 2)).abs()  # n x 7 or 8
                lower_bound = mean - interval
                upper_bound = mean + interval
                # 0 or 1  # keep 7 or 8
                valid_mask = (gt_boxes > lower_bound).float() * (
                        gt_boxes < upper_bound
                ).float()

                # if valid_mask.any():
                picp_cnt = valid_mask.sum(dim=0)
                mpiw_sum = (2 * interval * valid_mask).sum(dim=0)
                pred_cnt = box_preds.shape[0]

                recall_dict["picp_cnt"] += picp_cnt
                recall_dict["mpiw_sum"] += mpiw_sum
                recall_dict["pred_cnt"] += pred_cnt
                recall_dict["uncer_type"] = uncer_type

        return recall_dict
