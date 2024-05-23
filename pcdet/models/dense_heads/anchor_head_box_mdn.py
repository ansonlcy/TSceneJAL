import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from .anchor_head_template import AnchorHeadTemplate
from ...utils import box_coder_utils, common_utils, loss_utils


class AnchorHeadBoxMdn(AnchorHeadTemplate):
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

        var_channel = input_channels
        # add mlp to reduce the channel (384 -> 64)
        if self.model_cfg.get('USE_DOWNSAMPLE', None) is True:
            var_channel = input_channels // 4
            self.mlp = MLP(in_channel=input_channels, out_channel=var_channel, hidden=input_channels // 2)

        for idx in range(1, self.num_gauss + 1):
            # use (x,y,z,l,w,h,theta) to represent a box, so has 7 var
            setattr(self, 'box_var_{}'.format(idx), nn.Conv2d(
                var_channel, self.num_anchors_per_location * 7,
                kernel_size=1
            ))
            setattr(self, 'box_pi_{}'.format(idx), nn.Conv2d(
                var_channel, self.num_anchors_per_location * 7,
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
        data_dict['rpn_cls_preds'] = cls_preds

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
            batch_cls_preds, batch_box_preds, batch_box_preds_var, batch_box_preds_pi, batch_box_preds_raw = self.generate_predicted_boxes_mdn(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=self.forward_ret_dict['mean_preds'], var_preds=self.forward_ret_dict['var_preds'], pi_preds=self.forward_ret_dict['pi_preds'],
                dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['batch_box_preds_var'] = batch_box_preds_var
            data_dict['batch_box_preds_pi'] = batch_box_preds_pi
            data_dict['batch_box_preds_raw'] = batch_box_preds_raw  # this is to cal eu

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

        # mdn loss

        # nll loss
        # mean [batch_size, H, W, 6 * 7, num_gauss]
        # var  [batch_size, H, W, 6 * 7, num_gauss]
        # pi   [batch_size, H, W, 6 * 7, num_gauss]

        for name in ["mean", "pi", "var"]:
            s = self.forward_ret_dict[f'{name}_preds'].shape[-2]
            var_names[f'pos_{name}_preds'] = self.forward_ret_dict[f'{name}_preds'].view(batch_size, -1,

                                                                                                     s // self.num_anchors_per_location, self.num_gauss)[positives]

        # add sin difference to the mdn out
        pos_box_preds_sin_list = []
        pos_reg_targets_sin_list = []
        pos_box_reg_targets = box_reg_targets[positives]
        for idx in range(self.num_gauss):
            pos_box_preds = var_names[f'pos_mean_preds'][..., idx]
            pos_box_preds_sin, pos_reg_targets_sin = self.add_sin_difference(pos_box_preds, pos_box_reg_targets)
            pos_box_preds_sin_list.append(pos_box_preds_sin.unsqueeze(-1))
            pos_reg_targets_sin_list.append(pos_reg_targets_sin.unsqueeze(-1))

        pos_box_preds_sin = torch.cat(pos_box_preds_sin_list, dim=-1)
        pos_reg_targets_sin = torch.cat(pos_reg_targets_sin_list, dim=-1)
        
        box_mdn_loss = loss_utils.box_mdn_loss(pos_box_preds_sin, pos_reg_targets_sin, var_names['pos_var_preds'], var_names['pos_pi_preds'])
        loc_mdn_loss = box_mdn_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_mdn_weight']
        tb_dict = {
            'loc_mdn_loss': loc_mdn_loss.item()
        }
        
        box_loss = loc_mdn_loss


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

    def generate_predicted_boxes_mdn(self, batch_size, cls_preds, box_preds, var_preds, pi_preds, dir_cls_preds=None):
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

        box_preds_sum = torch.sum(box_preds * pi_preds, dim=-1)

        batch_box_preds = box_preds_sum.view(batch_size, num_anchors, -1) if not isinstance(box_preds_sum, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        # the batch_box_preds_raw is the out put of mdn, this is to cal eu
        batch_box_preds_raw_list = []
        batch_box_pred_raw = box_preds.view(batch_size, num_anchors, -1, self.num_gauss)
        for i in range(self.num_gauss):
            batch_box_preds_raw_list.append(self.box_coder.decode_torch(batch_box_pred_raw[:, :, :, i], batch_anchors).unsqueeze(-1))
        batch_box_pred_raw = torch.cat(batch_box_preds_raw_list, dim=-1)



        if self.model_cfg.get('UNCER_TYPE', None) == 'center':
            batch_box_preds_var = var_preds.view(batch_size, num_anchors, -1, self.num_gauss)

        batch_box_preds_pi = pi_preds.view(batch_size, num_anchors, -1, self.num_gauss)


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

            # process the batch_box_pred_raw
            dir_rot_raw = common_utils.limit_period(
                batch_box_pred_raw[:, :, 6, :] - dir_offset, dir_limit_offset, period
            )
            batch_box_pred_raw[:, :, 6, :] = dir_rot_raw + dir_offset + period * dir_labels.unsqueeze(-1).repeat(1, 1, self.num_gauss).to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds, batch_box_preds_var, batch_box_preds_pi, batch_box_pred_raw


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

