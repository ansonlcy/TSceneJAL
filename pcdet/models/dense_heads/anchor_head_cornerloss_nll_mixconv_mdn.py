import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from .anchor_head_template import AnchorHeadTemplate
from ...utils import box_coder_utils, common_utils, loss_utils


class AnchorHeadCornerNllMixMdn(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )

        # mean
        assert self.model_cfg.get('NUM_GAUSS', None) is not None, "NUM_GAUSS shouldn't be None"
        self.num_gauss = self.model_cfg.get('NUM_GAUSS', None)

        for idx in range(1, self.num_gauss + 1):
            setattr(self, 'box_mean_{}'.format(idx), nn.Conv2d(
                input_channels, self.num_anchors_per_location * self.box_coder.code_size,
                kernel_size=1
            ))


        # varance and pi
        assert (self.model_cfg.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION and self.model_cfg.LOSS_CONFIG.CENTER_LOSS_REGULARIZATION) == False, \
            "CORNER_LOSS and CENTER_LOSS shouldn't be True at the same time"
        var_channel = input_channels
        # add mlp to reduce the channel (384 -> 64)
        if self.model_cfg.get('USE_DOWNSAMPLE', None) is True:
            var_channel = input_channels // 4
            self.mlp = MLP(in_channel=input_channels, out_channel=var_channel, hidden=input_channels // 2)

        if self.model_cfg.get('UNCER_TYPE', None) == 'corner':
            # one box has 8 corners
            if self.model_cfg.get('SEPERATE_VAR', None) is True:
                sep_num = 3
            else:
                sep_num = 1

            for idx in range(1, self.num_gauss + 1):
                setattr(self, 'box_var_{}'.format(idx), nn.Conv2d(
                    var_channel, self.num_anchors_per_location * 8 * sep_num,
                    kernel_size=1
                ))
                setattr(self, 'box_pi_{}'.format(idx), nn.Conv2d(
                    var_channel, self.num_anchors_per_location * 8 * sep_num,
                    kernel_size=1
                ))


        elif self.model_cfg.get('UNCER_TYPE', None) == 'center':
            # use (x,y,z,l,w,h,theta) to represent a box, so has 7 var
            for idx in range(1, self.num_gauss + 1):
                setattr(self, 'box_var_{}'.format(idx), nn.Conv2d(
                    var_channel, self.num_anchors_per_location * 7,
                    kernel_size=1
                ))
                setattr(self, 'box_pi_{}'.format(idx), nn.Conv2d(
                    var_channel, self.num_anchors_per_location * 8,
                    kernel_size=1
                ))

        else:
            raise NotImplementedError


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
        for idx in range(1, self.num_gauss + 1):
            nn.init.normal_(getattr(self, 'box_mean_{}'.format(idx)).weight, mean=0, std=0.001)
            nn.init.xavier_normal_(getattr(self, 'box_var_{}'.format(idx)).weight)
            nn.init.xavier_normal_(getattr(self, 'box_pi_{}'.format(idx)).weight)

    def forward(self, data_dict):
        v = locals()

        spatial_features_2d = data_dict['spatial_features_2d']
        # box var feature
        var_feature = spatial_features_2d
        if self.model_cfg.get('USE_DOWNSAMPLE', None) is True:
            var_feature = self.mlp(var_feature)

        # class preds
        cls_preds = self.conv_cls(spatial_features_2d)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        self.forward_ret_dict['cls_preds'] = cls_preds

        # box direction class preds
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        for idx in range(1, self.num_gauss + 1):
            v[f'box_mean_preds_{idx}'] = getattr(self, f'box_mean_{idx}')(spatial_features_2d).permute(0, 2, 3,
                                                                                                       1).contiguous()

            v[f'box_var_preds_{idx}'] = F.elu(getattr(self, f'box_var_{idx}')(var_feature)) + 1 + 1e-8
            v[f'box_var_preds_{idx}'] = v[f'box_var_preds_{idx}'].permute(0, 2, 3, 1).contiguous()

            v[f'box_pi_preds_{idx}'] = getattr(self, f'box_pi_{idx}')(var_feature).permute(0, 2, 3, 1).contiguous()

        self.forward_ret_dict['mean_preds'] = torch.cat(
            [v[f'box_mean_preds_{idx}'].unsqueeze(-1) for idx in range(1, self.num_gauss + 1)], dim=-1)
        self.forward_ret_dict['var_preds'] = torch.cat(
            [v[f'box_var_preds_{idx}'].unsqueeze(-1) for idx in range(1, self.num_gauss + 1)], dim=-1)
        self.forward_ret_dict['pi_preds'] = torch.cat(
            [v[f'box_pi_preds_{idx}'].unsqueeze(-1) for idx in range(1, self.num_gauss + 1)], dim=-1)

        # the sum of pi need to be 1
        self.forward_ret_dict['pi_preds'] = torch.softmax(self.forward_ret_dict['pi_preds'], dim=-1)


        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds, batch_box_preds_var = self.generate_predicted_boxes_mdn(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=self.forward_ret_dict['mean_preds'], var_preds=self.forward_ret_dict['var_preds'],
                dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['batch_box_preds_var'] = batch_box_preds_var

            data_dict['cls_preds_normalized'] = False

        return data_dict

    def get_box_reg_layer_loss(self):

        var_names = locals()
        # box_preds = self.forward_ret_dict['mean_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        # batch_size = int(box_preds.shape[0])
        batch_size = int(self.forward_ret_dict['mean_preds'].shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

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


        # origial smooth l1 loss
        box_preds = self.forward_ret_dict['mean_preds'].mean(dim=-1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        # nll loss
        # mean [batch_size, H, W, 6 * 7, num_gauss]
        # var  [batch_size, H, W, 6 * (7 or 8), num_gauss]
        # pi   [batch_size, H, W, 6 * (7 or 8), num_gauss]

        for name in ["mean", "pi", "var"]:
            s = self.forward_ret_dict[f'{name}_preds'].shape[-2]
            var_names[f'pos_{name}_preds'] = self.forward_ret_dict[f'{name}_preds'].view(batch_size, -1,
                                                                                                     s // self.num_anchors_per_location, self.num_gauss)[
                    positives]


        if self.model_cfg.get('UNCER_TYPE', None) == 'corner':

            if self.model_cfg.get('SEPERATE_VAR', None) is True:
                var_names[f'pos_var_preds'] = var_names[f'pos_var_preds'].unsqueeze(-2).view(var_names[f'pos_var_preds'].shape[0], 8, 3, self.num_gauss)
                var_names[f'pos_pi_preds'] = var_names[f'pos_pi_preds'].unsqueeze(-2).view(var_names[f'pos_pi_preds'].shape[0], 8, 3, self.num_gauss)
            else:
                var_names[f'pos_var_preds'] = var_names[f'pos_var_preds'].unsqueeze(-2).expand(-1, -1, 3, -1)
                var_names[f'pos_pi_preds'] = var_names[f'pos_pi_preds'].unsqueeze(-2).expand(-1, -1, 3, -1)

        elif self.model_cfg.get('UNCER_TYPE', None) == 'center':
            box_var_preds_pos = var_names[f'pos_var_preds'].expand(-1, 7, self.num_gauss)
            box_var_preds_pos = var_names[f'pos_pi_preds'].expand(-1, 7, self.num_gauss)

        else:
            raise NotImplementedError

        if self.model_cfg.LOSS_CONFIG.CENTER_LOSS_REGULARIZATION and len(torch.nonzero(positives)) > 0:
            # center nll loss
            box_preds_center = box_preds[positives]
            box_reg_targets_center = box_reg_targets[positives]
            # the varance is base to gt not target (or use target varance but need to do error backprop)
            box_preds_gt_center = self.box_coder.decode_torch(box_preds_center, anchors[positives])
            box_reg_gt_center = self.box_coder.decode_torch(box_reg_targets_center, anchors[positives])

            center_nll_loss = loss_utils.nll(box_preds_gt_center, box_reg_gt_center, box_var_preds_pos)
            center_nll_loss = center_nll_loss.mean()
            center_nll_loss = center_nll_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['center_var_loss_weight']
            box_loss += center_nll_loss
            tb_dict['center_var_loss'] = center_nll_loss.item()

        if self.model_cfg.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION and len(torch.nonzero(positives)) > 0:
            # corner loss + nll loss
            box_preds_3d = self.box_coder.decode_torch(box_preds, anchors)
            box_gt_3d = self.box_coder.decode_torch(box_reg_targets, anchors)

            # corner loss
            box_gt_3d_pos = box_gt_3d[positives]
            box_preds_3d_pos = box_preds_3d[positives]
            box_loss_corner = loss_utils.get_corner_loss_lidar(box_preds_3d_pos, box_gt_3d_pos)

            box_loss_corner = box_loss_corner.mean()
            box_loss_corner = box_loss_corner * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['corner_loss_weight']
            box_loss += box_loss_corner
            tb_dict['corner_loss'] = box_loss_corner.item()

            # corner mdn nll loss
            box_loss_corner_mdn = loss_utils.corner_mdn_loss(var_names['pos_mean_preds'], box_gt_3d_pos, var_names['pos_var_preds'], var_names['pos_pi_preds'])
            box_loss_corner_var = box_loss_corner_mdn.mean()
            box_loss_corner_var = box_loss_corner_var * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS[
                'corner_var_loss_weight']
            box_loss += box_loss_corner_var
            tb_dict['corner_var_loss'] = box_loss_corner_var.item()

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

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def generate_predicted_boxes_mdn(self, batch_size, cls_preds, box_preds, var_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            var_preds: dict {var_preds:(N, H, W, 6*(8*3))} or
                            {var_preds:(N, H, W, 6*8)}
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
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if self.model_cfg.get('UNCER_TYPE', None) == 'corner':
            # box_var
            if self.model_cfg.get('SEPERATE_VAR', None) is True:
                batch_box_preds_var = var_preds['var_preds'].view(batch_size, -1, 8,
                                                                  3)  # 8 corners, each corner has 3 var(x_var, y_var, z_var)
            else:
                batch_box_preds_var = var_preds['var_preds'].view(batch_size, -1, 8,
                                                                  1)  # 8 corners, each corner has 1 var(x_var=y_var=z_var)
                batch_box_preds_var = batch_box_preds_var.expand(-1, -1, -1, 3)

        if self.model_cfg.get('UNCER_TYPE', None) == 'center':
            batch_box_preds_var = var_preds['var_preds'].view(batch_size, -1, 7)

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

        return batch_cls_preds, batch_box_preds, batch_box_preds_var


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, hidden, bias=False):
        super(MLP, self).__init__()

        # insert the layers

        self.linear1 = nn.Conv2d(in_channel, hidden, kernel_size=1)
        self.linear1.apply(self._init_weights)
        self.linear2 = nn.Conv2d(hidden, out_channel, kernel_size=1)
        self.linear2.apply(self._init_weights)

        self.norm1 = nn.BatchNorm2d(hidden)
        self.norm2 = nn.BatchNorm2d(out_channel)

        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.linear1(x)
        torch.backends.cudnn.enabled = False
        x = self.norm1(x)
        torch.backends.cudnn.enabled = True
        x = self.act1(x)
        x = self.linear2(x)
        torch.backends.cudnn.enabled = False
        x = self.norm2(x)
        torch.backends.cudnn.enabled = True

        return self.act2(x)

