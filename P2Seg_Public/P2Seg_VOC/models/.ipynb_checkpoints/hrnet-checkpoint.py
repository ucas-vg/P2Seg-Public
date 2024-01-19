# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hrnet_config import cfg
from models.hrnet_config import update_config
import time

# import numpy as np

from utils.utils import refine_label_generation, refine_label_generation_with_point

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, heads, args, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNet, self).__init__()

        self.heads = heads
        self.args = args
        
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        for head in self.heads:
            classes = self.heads[head]
            head_conv = 256
            
            fc = nn.Sequential(
                nn.Conv2d(pre_stage_channels[0], head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, classes, kernel_size=1, stride=1, padding=0, bias=True),
            )

            self.__setattr__(head, fc)

        self.aff_conv = nn.Conv2d(48, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.mask_pooling = torch.nn.MaxPool2d(kernel_size=4, stride=4)
        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, seg_map=None, label=None, point_list=None, masks=None, target_shape=None):
        if target_shape is None:
            target_shape = x.shape[-2:]

        # print(x.shape, 'xxxxxxxxxxx')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        results = {}
        # begin = time.time()
        # ####################################### for affinity
        if self.args.affinity_loss:
            aff = self.aff_conv(y_list[0])
            # aff = torch.nn.functional.interpolate(aff,
            #                                     size=target_shape,
            #                                     mode='bilinear',
            #                                     align_corners=False)
            aff = aff.view(aff.size(0), aff.size(1), -1)
            aff = torch.matmul(aff.transpose(-1, -2), aff)# 矩阵乘HXW，关系矩阵得到的东西做哈达玛
            #aff = aff * aff * aff * aff
            aff = torch.sigmoid(aff)

            if seg_map is not None:
                masks = self.mask_pooling(masks.float())
                aff_label = self.mask_to_aff(masks)
                # aff_label = torch.zeros((masks.shape[-1] * masks.shape[-2], masks.shape[-1] * masks.shape[-2])).cuda()
                aff_loss = self.aff_loss(aff, aff_label)

                results['affinity_1'] = aff_loss
                # results['affinity'] = torch.tensor([0.]).cuda()
        ########################################

        for head in self.heads:
            results[head] = self.__getattr__(head)(y_list[0])
            # ############## for affinity matmult to segmap #######################################
            if self.args.affinity_loss and head == 'seg':
                for i in range(results[head].shape[0]):
                    B, C, H, W = results[head].shape
                    _seg = results[head][i].reshape(results[head].size(1), -1)
#                     print(label.shape, label, i, 'B:', B, C, H,W)
                    valid_key = torch.nonzero(label[i, ...])[:, 0]
                    _seg = _seg[valid_key, ...]
                    _seg = F.softmax(_seg, dim=0)
                    # _aff = F.softmax(aff[i], dim=1)
                    _aff = torch.div(aff[i], torch.sum(aff[i], dim=1).unsqueeze(1))
                    _seg_rw = torch.matmul(_seg, _aff)
                    results[head][i, valid_key,:] = _seg_rw.reshape(-1, H, W)

            ######################################################
            results[head] = torch.nn.functional.interpolate(results[head], 
                                                            size=target_shape, 
                                                            mode='bilinear', 
                                                            align_corners=False)
        # end = time.time()
        # print('t: ', end-begin)
        if self.args.annotation_loss and seg_map is not None:
            all_pred = []
            all_label = []
            for b in range(results['seg'].shape[0]):
                _valid_cls = torch.nonzero(label[b])[0]
                _point_list = point_list[b]
                _pred = []
                _label = []
                for cls in _valid_cls:
                    for gt_y, gt_x in _point_list[cls]:
                        if gt_y != 0 and gt_x != 0:
                            _pred.append(results['seg'][b,:,gt_y,gt_x])
                            _label.append(cls)
                if  len(_pred) > 0:
                    all_pred.append(torch.stack(_pred))
                    all_label.append(torch.stack(_label))
            if len(all_pred) > 0:
                all_pred = torch.stack(all_pred)
                all_label = torch.stack(all_label)
                results['ann_loss'] = self.ce_loss(all_pred.reshape(-1, all_pred.shape[-1]), all_label.reshape(-1))

        if seg_map is not None: # refined label generation
            if self.args.sup == 'point': # point supervision setting
                pseudo_label = refine_label_generation_with_point(
                    results['seg'].clone().detach(), 
                    point_list.cpu().numpy(), 
                    results['offset'].clone().detach(), 
                    label.clone().detach(), 
                    seg_map.clone().detach(),
                    self.args,
                )

                # ############## for affinity iteration #######################################
                if self.args.affinity_loss:
                    if self.args.iterate_refine == 1:
                        masks = self.mask_pooling(pseudo_label['masks'])
                        aff_label = self.mask_to_aff(masks.unsqueeze(1))
                        # aff_label = torch.zeros((masks.shape[-1] * masks.shape[-2], masks.shape[-1] * masks.shape[-2])).cuda()
                        aff_loss = self.aff_loss(aff, aff_label)
                        results['affinity_2'] = aff_loss
                # ############## for affinity iteration #######################################
            else:  # image-level supervision setting
                pseudo_label = refine_label_generation(
                    results['seg'].clone().detach(), 
                    results['center'].clone().detach(), 
                    results['offset'].clone().detach(), 
                    label.clone().detach(), 
                    seg_map.clone().detach(),
                    self.args,
                )
            
            return results, pseudo_label
        
        return results

    def aff_loss(self, pred_aff, label_aff):
        '''
        pred_aff, label_aff: b, h*w, h*w
        '''
        pos_label = (label_aff == 1).type(torch.int16)
        pos_count = pos_label.sum() + 1
        neg_label = (label_aff == 0).type(torch.int16)
        neg_count = neg_label.sum() + 1
        pred_aff = torch.sigmoid(input=pred_aff)

        pos_loss = torch.sum(pos_label * (1 - pred_aff)) / pos_count
        neg_loss = torch.sum(neg_label * (pred_aff)) / neg_count

        return 0.5 * pos_loss + 0.5 * neg_loss

    def mask_to_aff(self, mask):
        """
        mask: array(b, c, h, w)
        """
        # begin = time.time()
        N, C, H, W = mask.shape
        mask = mask.reshape(N, C, -1)  # (b, -1)
        mask = torch.matmul(mask.transpose(-1, -2), mask)

        # out = torch.matmul(mask, mask.permute(0, 2, 1))

        mask = torch.where(torch.sum(mask, dim=0) > 0, 1, 0)

        # print('t: ', time.time() - begin)
        # out = out.cpu()

        return mask
    
    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            print('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        else:
            logger.info("=> without pre-trained models")
            print("=> without pre-trained models")


def HRNet(layer, heads, args, **kwargs):
    update_config(cfg, f"./models/hrnet_config/w{layer}_384x288_adam_lr1e-3.yaml")
    cfg.MODEL.NUM_JOINTS = 1 # num classes
    
    model = PoseHighResolutionNet(cfg, heads, args, **kwargs)
    
    model.init_weights(cfg.MODEL.PRETRAINED)
    
    return model


def hrnet32(args):
    heads = {
        'seg': args.num_classes+1, 
        'center': args.num_classes, 
        'offset': 2
    }
    model = HRNet(32, heads, args)
    return model
        
def hrnet48(args):
    heads = {
        'seg': args.num_classes+1, 
        'center': args.num_classes, 
        'offset': 2
    }
    model = HRNet(48, heads, args)
    return model