import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from .anchor_head_template import AnchorHeadTemplate
from ...utils import box_coder_utils, common_utils, loss_utils
from torch.distributions import OneHotCategorical


class AnchorHeadSingleMdnTp5(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.num_gauss = self.model_cfg.get('NUM_GAUSS')  # the number of gauss in mdn

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )

        # The mu/sigma net (not include theta)
        for idx in range(1, self.num_gauss + 1):
            self.add_module(f'conv_box_mu_{idx}', nn.Conv2d(
                input_channels, self.num_anchors_per_location * (self.box_coder.code_size - 1),
                kernel_size=1
            ))

            self.add_module(f'conv_box_sigma_{idx}', nn.Conv2d(
                    input_channels, self.num_anchors_per_location * (self.box_coder.code_size - 1),
                    kernel_size=1
                ))

        self.add_module(f'conv_box_pi', nn.Conv2d(
                    input_channels, self.num_anchors_per_location * (self.box_coder.code_size - 1) * self.num_gauss,
                    kernel_size=1
                ))

        # The theta net
        self.add_module(f'conv_box_theta', nn.Conv2d(
            input_channels, self.num_anchors_per_location * 1,
            kernel_size=1
        ))

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))


    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        self.forward_ret_dict['cls_preds'] = cls_preds

        self.forward_ret_dict['box_preds'] = {}

        box_preds_theta = getattr(self, f'conv_box_theta')(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
        self.forward_ret_dict['box_preds'][f'box_preds_theta'] = box_preds_theta

        var_names = locals()
        for idx in range(1, self.num_gauss + 1):
            var_names[f'box_preds_mu_{idx}'] = getattr(self, f'conv_box_mu_{idx}')(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['box_preds'][f'box_preds_mu_{idx}'] = var_names[f'box_preds_mu_{idx}']

            var_names[f'box_preds_sigma_{idx}'] = getattr(self, f'conv_box_sigma_{idx}')(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
            # sigma needs to be positive
            var_names[f'box_preds_sigma_{idx}'] = F.elu(var_names[f'box_preds_sigma_{idx}']) + 1
            self.forward_ret_dict['box_preds'][f'box_preds_sigma_{idx}'] = var_names[f'box_preds_sigma_{idx}']

        # predict the pi, and do the process to make that sum pi = 1
        box_preds_pi = torch.unsqueeze(getattr(self, f'conv_box_pi')(spatial_features_2d).permute(0, 2, 3, 1).contiguous(), -1)
        box_preds_pi = box_preds_pi.view(box_preds_pi.size(0),box_preds_pi.size(1),box_preds_pi.size(2),-1,self.num_gauss)
        box_preds_pi = OneHotCategorical(logits=box_preds_pi).logits

        # split the pi
        for idx in range(1, self.num_gauss + 1):
            self.forward_ret_dict['box_preds'][f'box_preds_pi_{idx}'] = box_preds_pi[..., idx-1]

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=self.forward_ret_dict['box_preds'], dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

            data_dict['gauss_output'] = {}
            for idx in range(1, self.num_gauss+1):
                for name in ['mu', 'sigma', 'pi']:
                    data_dict['gauss_output'][f'box_preds_{name}_{idx}'] = self.forward_ret_dict['box_preds'][f'box_preds_{name}_{idx}'].view(data_dict['batch_size'],-1,6)


        return data_dict
    
    
    def get_box_reg_layer_mdn_loss(self):

        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_reg_targets.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        var_names = locals()
        box_preds = self.forward_ret_dict['box_preds']

        
        for idx in range(1, self.num_gauss + 1):
            s = box_preds[f'box_preds_sigma_{idx}'].shape[-1]
            var_names[f'pos_box_preds_sigma_{idx}'] = box_preds[f'box_preds_sigma_{idx}'].view(batch_size, -1,
                                                                                                     s // self.num_anchors_per_location if not self.use_multihead else s)[
                    positives]
        
            var_names[f'pos_box_preds_mu_{idx}'] = box_preds[f'box_preds_mu_{idx}'].view(batch_size, -1,
                                                                                                     s // self.num_anchors_per_location if not self.use_multihead else s)[
                    positives]
        
        
            var_names[f'pos_box_preds_pi_{idx}'] = box_preds[f'box_preds_pi_{idx}'].view(batch_size, -1, s // self.num_anchors_per_location if not self.use_multihead else s)[positives]

        pos_box_preds_theta = box_preds[f'box_preds_theta'].view(batch_size, -1, 1)[positives]
        pos_box_pred = {}
        for name in ['mu', 'sigma', 'pi']:
            for idx in range(1, self.num_gauss + 1):
                pos_box_pred[f'pos_box_preds_{name}_{idx}'] = var_names[f'pos_box_preds_{name}_{idx}']

        pos_box_pred['pos_box_preds_theta'] = pos_box_preds_theta

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)

        # box_preds_sin, reg_targets_sin = self.add_sin_difference(pos_box_pred, box_reg_targets[positives])
        # loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin)
        weight_conf = self.model_cfg.LOSS_CONFIG.get('LOSS_WEIGHTS')
        loc_loss_src = self.reg_loss_func(pos_box_pred, box_reg_targets[positives], weight_conf)
        loc_loss = loc_loss_src / positives.float().sum()

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_mdn_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict


    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: dict
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)

        _box_preds = torch.zeros_like(box_preds['box_preds_mu_1'].view(-1, self.box_coder.code_size-1))
        for idx in range(1, self.num_gauss + 1):
            pi = torch.exp(box_preds[f'box_preds_pi_{idx}'])
            _box_preds += (pi * box_preds[f'box_preds_mu_{idx}']).view(-1, self.box_coder.code_size-1)

        _box_preds = torch.cat((_box_preds, box_preds['box_preds_theta'].view(-1, 1)), dim=-1)


        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = _box_preds.view(batch_size, num_anchors, -1) if not isinstance(_box_preds, list) \
            else torch.cat(_box_preds, dim=1).view(batch_size, num_anchors, -1)


        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        # proess the box_preds
        # for idx in range(1, self.num_gauss+1):
        #     box_preds[f'box_preds_mu_{idx}'] = box_preds[f'box_preds_mu_{idx}'].view(batch_size, num_anchors, -1)
        #     box_preds[f'box_preds_s_{idx}'] = box_preds[f'box_preds_s_{idx}'].view(batch_size, num_anchors, -1)
        #     box_preds[f'box_preds_pi_{idx}'] = pi_all[idx - 1].view(batch_size, num_anchors, -1)

        return batch_cls_preds, batch_box_preds
