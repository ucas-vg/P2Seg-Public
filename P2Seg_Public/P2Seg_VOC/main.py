"""
BESTIE
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import numpy as np
import torch
import argparse
import os
import cv2
import time
import random
import pickle
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from models import model_factory
from utils.LoadData import get_dataset_voc, get_dataset_coco
from utils.my_optim import WarmupPolyLR
from utils.loss import Weighted_L1_Loss, Weighted_MSELoss, DeepLabCE
from utils.utils import AverageMeter, get_ins_map, get_ins_map_with_point, get_ins_map_with_point_forallpoint

import chainercv
from chainercv.datasets import VOCInstanceSegmentationDataset
from chainercv.evaluations import eval_instance_segmentation_voc
from torchinfo import summary


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def mask_to_aff(mask):
    """
    mask: array(b, h, w)
    """
    # begin = time.time()
    N = mask.shape[0]
    mask = mask.reshape(N, -1)  # (b, -1)
    mask = mask.unsqueeze(-1)

    out = torch.matmul(mask, mask.permute(0, 2, 1))

    out = torch.sum(out, dim=0)

    out = torch.where(out > 0, 1, 0)

    # print('t: ', time.time() - begin)
    # out = out.cpu()

    return out


def parse():
    parser = argparse.ArgumentParser(description='BESTIE pytorch implementation')
    parser.add_argument("--root_dir", type=str, default='data/VOCdevkit/VOC2012', help='Root dir for the project')
    parser.add_argument('--sup', type=str, help='supervision source', choices=["cls", "point"], default='point')
    parser.add_argument("--dataset", type=str, default='voc', choices=["voc", "coco"])
    parser.add_argument("--backbone", type=str, default='hrnet48',
                        choices=["resnet50", "resnet101", "hrnet34", "hrnet48"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=416)
    parser.add_argument("--num_classes", type=int, default=20, help='you must choose a number of classes from [20, 80]')
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--train_iter", type=int, default=50000)
    parser.add_argument("--warm_iter", type=int, default=2000, help='warm-up iterations')
    parser.add_argument("--train_epoch", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument('--resume', default=None, type=str, help='weight restore')

    parser.add_argument('--save_folder', default='checkpoints/test1', help='Location to save checkpoint models')
    parser.add_argument('--print_freq', default=200, type=int, help='interval of showing training conditions')
    parser.add_argument('--save_freq', default=10000, type=int, help='interval of save checkpoint models')
    parser.add_argument("--cur_iter", type=int, default=0, help='current training interations')

    parser.add_argument("--gamma", type=float, default=0.9, help='learning rate decay power')
    parser.add_argument("--pseudo_thresh", type=float, default=0.7, help='threshold for pseudo-label generation')
    parser.add_argument("--refine_thresh", type=float, default=0.3, help='threshold for refined-label generation')
    parser.add_argument("--kernel", type=int, default=41, help='kernel size for point extraction')
    parser.add_argument("--sigma", type=int, default=6, help='sigma of 2D gaussian kernel')
    parser.add_argument("--beta", type=float, default=3.0, help='parameter for center-clustering')
    parser.add_argument("--bn_momentum", type=float, default=0.01)
    parser.add_argument('--refine', type=str2bool, default=True, help='enable self-refinement.')
    parser.add_argument("--refine_iter", type=int, default=0, help='self-refinement running iteration')
    parser.add_argument("--seg_weight", type=float, default=1.0, help='loss weight for segmantic segmentation map')
    parser.add_argument("--center_weight", type=float, default=200.0, help='loss weight for center map')
    parser.add_argument("--offset_weight", type=float, default=0.01, help='loss weight for offset map')

    parser.add_argument('--val_freq', default=1000, type=int, help='interval of model validation')
    parser.add_argument("--val_thresh", type=float, default=0.1,
                        help='threhsold for instance-groupping in validation phase')
    parser.add_argument("--val_kernel", type=int, default=41,
                        help='kernsl size for point extraction in validation phase')
    parser.add_argument('--val_flip', type=str2bool, default=False,
                        help='enable flip test-time augmentation in vadliation phase')
    parser.add_argument('--val_clean', type=str2bool, default=True,
                        help='cleaning pseudo-labels using image-level labels')
    parser.add_argument('--val_ignore', type=str2bool, default=False, help='ignore')

    parser.add_argument("--random_seed", type=int, default=3407)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    # parser.add_argument("--local_rank", type=int, default=int(os.environ["LOCAL_RANK"]))
    parser.add_argument("--local_rank", type=int, default=0)

    # parser.add_argument("--use_aff", type=str2bool, default=True)
    # parser.add_argument("--use_dis_group", type=str2bool, default=True)
    # parser.add_argument("--use_aff_refine", type=str2bool, default=True)
    # parser.add_argument("--use", type=str2bool, default=True)

    # flag add by vg
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--result_dir", type=str, default="checkpoints/results_tmp")
    parser.add_argument("--save_to_rle", type=str2bool, default=True)
    parser.add_argument("--dis_group", type=str2bool, default=False)
    parser.add_argument("--affinity_loss", type=str2bool, default=False)
    parser.add_argument("--annotation_loss", type=str2bool, default=False)
    parser.add_argument("--iterate_refine", type=int, default=0)
    parser.add_argument("--all_point", type=str2bool, default=True)

    # dynamic distance group
    # inference with distance group

    return parser.parse_args()


def print_func(string):
    if torch.distributed.get_rank() == 0:
        print(string)


def save_checkpoint(save_path, model):
    if torch.distributed.get_rank() == 0:
        print('\nSaving state: %s\n' % save_path)
        state = {
            'model': model.module.state_dict(),
        }
        torch.save(state, save_path)
        
def calculate_and_print_gflops_train(model, x, device):
    def calculate_gflops(model, input_size, device):
        model_info = summary(model, input_data=dict(x=x), verbose=0, device=device)
        gflops = model_info.total_mult_adds / 1e9
        return gflops
    target_size1 = (1, 3, 352, 512)  
    # 计算并打印GFLOPs
    GFLOPS = calculate_gflops(model, target_size1, device)
    return GFLOPS

def calculate_and_print_gflops_trainrefine(model, x, seg_map, input_label, point_list, masks, device):
    def calculate_gflops(model, input_size, input_label, device):
        model_info = summary(model, input_data=dict(x=x, seg_map=seg_map, label=input_label, point_list=point_list, masks=masks), verbose=0, device=device)
        gflops = model_info.total_mult_adds / 1e9
        return gflops
    target_size1 = (1, 3, 352, 512)  
    # 计算并打印GFLOPs
    GFLOPS = calculate_gflops(model, target_size1, input_label, device)
    return GFLOPS

def train():
    batch_time = AverageMeter()
    avg_total_loss = AverageMeter()
    avg_seg_loss = AverageMeter()
    avg_pseudo_center_loss = AverageMeter()
    avg_refine_center_loss = AverageMeter()
    avg_pseudo_offset_loss = AverageMeter()
    avg_refine_offset_loss = AverageMeter()
    avg_aff_loss = AverageMeter()
    avg_refine_aff_loss = AverageMeter()
    avg_ann_loss = AverageMeter()

    best_AP = -1

    model.train()
    start = time.time()
    end = time.time()
    epoch = 0
    gflops_list = []
    
    for cur_iter in range(1, args.train_iter + 1):

        try:
            img, label, seg_map, center_map, offset_map, weight, point_list, masks = next(data_iter)
        except Exception as e:
            print_func("   [LOADER ERROR] " + str(e))

            epoch += 1
            data_iter = iter(train_loader)
            img, label, seg_map, center_map, offset_map, weight, point_list, masks = next(data_iter)

            end = time.time()
            batch_time.reset()
            avg_total_loss.reset()
            avg_seg_loss.reset()
            avg_pseudo_center_loss.reset()
            avg_refine_center_loss.reset()
            avg_pseudo_offset_loss.reset()
            avg_refine_offset_loss.reset()
            avg_aff_loss.reset()
            avg_refine_aff_loss.reset()

            avg_aff_loss.reset()

            
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        seg_map = seg_map.to(device, non_blocking=True)
        center_map = center_map.to(device, non_blocking=True)
        offset_map = offset_map.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        
        
        # masks = mask_pooling(masks.float())
        # aff = mask_to_aff(masks)

        run_refine = args.refine and (cur_iter > args.refine_iter)

        if run_refine:
            out, c_label = model(img, seg_map, label, point_list, masks)
            GFLOPS = calculate_and_print_gflops_trainrefine(model, img, seg_map, label, point_list, masks, device)
            gflops_list.append(GFLOPS)
        else:
            out = model(img)
            GFLOPS = calculate_and_print_gflops_train(model, img, device)
            gflops_list.append(GFLOPS)
        seg_loss = criterion['seg'](out['seg'], seg_map) * args.seg_weight
        center_loss_1 = criterion['center'](out['center'], center_map, weight) * args.center_weight
        offset_loss_1 = criterion['offset'](out['offset'], offset_map, weight) * args.offset_weight
        aff_loss_1 = out.get('affinity_1', torch.tensor([0]).to(offset_loss_1.device))

        ann_loss = out.get('ann_loss', torch.tensor([0]).to(offset_loss_1.device))

        center_loss_2 = center_loss_1
        offset_loss_2 = offset_loss_1
        aff_loss_2 = aff_loss_1

        if run_refine and args.sup == 'cls':
            center_loss_2 = criterion['center'](out['center'], c_label['center'],
                                                c_label['weight']) * args.center_weight

        if run_refine:
            offset_loss_2 = criterion['offset'](out['offset'], c_label['offset'],
                                                c_label['weight']) * args.offset_weight
            aff_loss_2 = out.get('affinity_2', torch.tensor([0]).to(offset_loss_1.device))

        # print(aff)
        # print('t1: ', time.time() - end)

        # aff_loss_1 = aff_loss(out['affinity'], aff)
        # print(aff_loss_1)
        # print('t2: ', time.time()-end)
        ######################################
        # aff_loss_1 = criterion['aff'](out['aff'], aff)
        # aff_loss_2 = criterion['aff'](out['aff'], c_label['aff'])
        #######################################
        loss = seg_loss + (center_loss_1 + center_loss_2) * 0.5 + (offset_loss_1 + offset_loss_2) * 0.5 + (
                    aff_loss_1 + aff_loss_2) * 0.5 + ann_loss

        # compute gradient and backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        batch_time.update((time.time() - end))
        end = time.time()

        avg_total_loss.update(loss.item(), img.size(0))
        avg_seg_loss.update(seg_loss.item(), img.size(0))
        avg_pseudo_center_loss.update(center_loss_1.item(), img.size(0))
        avg_refine_center_loss.update(center_loss_2.item(), img.size(0))
        avg_pseudo_offset_loss.update(offset_loss_1.item(), img.size(0))
        avg_refine_offset_loss.update(offset_loss_2.item(), img.size(0))

        avg_aff_loss.update(aff_loss_1.item(), img.size(0))
        avg_refine_aff_loss.update(aff_loss_2.item(), img.size(0))

        avg_ann_loss.update(ann_loss.item(), img.size(0))

        if cur_iter % args.print_freq == 0:
            batch_time.synch(device)
            avg_total_loss.synch(device)
            avg_seg_loss.synch(device)
            avg_pseudo_center_loss.synch(device)
            avg_refine_center_loss.synch(device)
            avg_pseudo_offset_loss.synch(device)
            avg_refine_offset_loss.synch(device)

            avg_aff_loss.synch(device)
            avg_refine_aff_loss.synch(device)
            avg_ann_loss.synch(device)

            if args.local_rank == 0:
                print('Progress: [{0}][{1}/{2}] ({3:.1f}%, {4:.1f} min) | '
                      'Time: {5:.1f} ms | '
                      'Left: {6:.1f} min | '
                      'TotalLoss: {7:.4f} | '
                      'SegLoss: {8:.4f} | '
                      'AnnLoss: {9:.4f} | '
                      'AffLoss: {10:.4f} ({11:.4f} + {12:.4f}) | '
                      'centerLoss: {13:.4f} ({14:.4f} + {15:.4f}) | '
                      'OffsetLoss: {16:.4f} ({17:.4f} + {18:.4f}) '.format(
                    epoch, cur_iter, args.train_iter,
                    cur_iter / args.train_iter * 100, (end - start) / 60,
                    batch_time.avg * 1000, (args.train_iter - cur_iter) * batch_time.avg / 60,
                    avg_total_loss.avg, avg_seg_loss.avg, avg_ann_loss.avg,
                    avg_aff_loss.avg + avg_refine_aff_loss.avg,
                    avg_aff_loss.avg, avg_refine_aff_loss.avg,
                    avg_pseudo_center_loss.avg + avg_refine_center_loss.avg,
                    avg_pseudo_center_loss.avg, avg_refine_center_loss.avg,
                    avg_pseudo_offset_loss.avg + avg_refine_offset_loss.avg,
                    avg_pseudo_offset_loss.avg, avg_refine_offset_loss.avg,
                )
                )

        if args.local_rank == 0 and cur_iter % args.save_freq == 0:
            save_path = os.path.join(args.save_folder, 'last.pt')
            save_checkpoint(save_path, model)

        if cur_iter % args.val_freq == 0:
            val_score = validate(valid_loader, args)

            if args.local_rank == 0 and val_score['map'] > best_AP:
                best_AP = val_score['map']
                print('\n Best mAP50, iteration : %d, mAP50 : %.2f \n' % (cur_iter, best_AP))
                average_gflops = np.mean(gflops_list)
                print(f"Average Model GFLOPs: {average_gflops}")
                save_path = os.path.join(args.save_folder, 'best.pt')
                save_checkpoint(save_path, model)

        end = time.time()

    if args.local_rank == 0:
        print('\n training done')
        save_path = os.path.join(args.save_folder, 'last.pt')
        save_checkpoint(save_path, model)

    val_score = validate(valid_loader, args)

    if args.local_rank == 0 and val_score['map'] > best_AP:
        best_AP = val_score['map']
        print('\n Best mAP50, iteration : %d, mAP50 : %.2f \n' % (cur_iter, best_AP))
        average_gflops = np.mean(gflops_list)
        print(f"Average Model GFLOPs: {average_gflops}")
        model_file = os.path.join(args.save_folder, 'best.pt')
        save_checkpoint(model_file, model)

def calculate_and_print_gflops(model, x, input_label, device):
    def calculate_gflops(model, input_size, input_label, device):
        model_info = summary(model, input_data=dict(x=x, label=input_label), verbose=0, device=device)
        gflops = model_info.total_mult_adds / 1e9
        return gflops
    target_size1 = (1, 3, 352, 512)  
    # 计算并打印GFLOPs
    GFLOPS = calculate_gflops(model, target_size1, input_label, device)
    return GFLOPS
#     print(f"Model GFLOPs for representative input size {target_size1[2:]}: {gflops}")

def validate(data_loader, args):
    gflops_list = []
    model.eval()
#     first_batch = next(iter(valid_loader))
#     print("First batch structure:")
#     print(first_batch)
#     print("Types and shapes of each element in the batch:")
#     for item in first_batch:
#         print(type(item), (item.shape if hasattr(item, 'shape') else "Not a tensor"))
#     _, _, _, _, tsize = first_batch
    pred_seg_maps, pred_labels, pred_masks, pred_scores = [], [], [], []
    val_dir = args.result_dir
    if args.local_rank == 0:
        os.makedirs(val_dir, exist_ok=True)

    torch.distributed.barrier()
    if args.all_point:
        for img, cls_label, points, points_cls, fname, tsize in tqdm(valid_loader):
            target_size = int(tsize[0]), int(tsize[1])
#             print(cls_label)
            if args.val_flip:
                img = torch.cat([img, img.flip(-1)], dim=0)

            out = model(img.to(device), label=cls_label, target_shape=target_size)
            
            GFLOPS = calculate_and_print_gflops(model, img.to(device), cls_label, device)
            gflops_list.append(GFLOPS)
            
            if args.sup == 'point' and args.val_clean:
                pred_seg, pred_label, pred_mask, pred_score, ann_pts = get_ins_map_with_point_forallpoint(out,
                                                                                     cls_label,
                                                                                     points,
                                                                                     points_cls,
                                                                                     target_size,
                                                                                     device,
                                                                                     args)
            else:
                pred_seg, pred_label, pred_mask, pred_score = get_ins_map(out,
                                                                          cls_label,
                                                                          target_size,
                                                                          device,
                                                                          args)
                ann_pts = []

            #  add ###############################################
            if args.save_to_rle:
                # performance without rle: 0.4960591558139911, disk ~4.2G
                # performance with rle: 0.49605246197537084,   disk ~50M
                import pycocotools.mask as mask_util
                pred_rle = []
                for mask in pred_mask:
                    rle = mask_util.encode(np.array(mask[:, :, None], order='F', dtype='uint8'))[0]
                    rle["counts"] = rle["counts"].decode("utf-8")
                    pred_rle.append(rle)
                pred_mask = pred_rle
            ###############################################
            with open(f'{val_dir}/{fname[0]}.pickle', 'wb') as f:
                pickle.dump({
                    "ann_pts": ann_pts,
                    'pred_label': pred_label,  # attention: cls here start form 0, but in annotation start from 1
                    'pred_mask': pred_mask,
                    'pred_score': pred_score,
                }, f)
    else:
        for img, cls_label, points, fname, tsize in tqdm(data_loader):
            target_size = int(tsize[0]), int(tsize[1])

            if args.val_flip:
                img = torch.cat([img, img.flip(-1)], dim=0)

            out = model(img.to(device), label=cls_label, target_shape=target_size)

            if args.sup == 'point' and args.val_clean:
                pred_seg, pred_label, pred_mask, pred_score, ann_pts = get_ins_map_with_point(out,
                                                                                     cls_label,
                                                                                     points,
                                                                                     target_size,
                                                                                     device,
                                                                                     args)
            else:
                pred_seg, pred_label, pred_mask, pred_score = get_ins_map(out,
                                                                          cls_label,
                                                                          target_size,
                                                                          device,
                                                                          args)
                ann_pts = []

            #  add ###############################################
            if args.save_to_rle:
                # performance without rle: 0.4960591558139911, disk ~4.2G
                # performance with rle: 0.49605246197537084,   disk ~50M
                import pycocotools.mask as mask_util
                pred_rle = []
                for mask in pred_mask:
                    rle = mask_util.encode(np.array(mask[:, :, None], order='F', dtype='uint8'))[0]
                    rle["counts"] = rle["counts"].decode("utf-8")
                    pred_rle.append(rle)
                pred_mask = pred_rle
            ###############################################
            with open(f'{val_dir}/{fname[0]}.pickle', 'wb') as f:
                pickle.dump({
                    "ann_pts": ann_pts,
                    'pred_label': pred_label,  # attention: cls here start form 0, but in annotation start from 1
                    'pred_mask': pred_mask,
                    'pred_score': pred_score,
                }, f)

    torch.distributed.barrier()
    
    average_gflops = np.mean(gflops_list)
    print(f"Average Model GFLOPs: {average_gflops}")
    
    ap_result = {"ap": None, "map": None}

    if args.local_rank == 0:
        pred_masks, pred_labels, pred_scores = [], [], []

        for fname in ins_gt_ids:
            with open(f'{val_dir}/{fname}.pickle', 'rb') as f:
                dat = pickle.load(f)
                pred_mask = dat['pred_mask']
                # #################################################################
                import pycocotools.mask as mask_util
                if args.save_to_rle:
                    pred_mask = np.array(mask_util.decode(pred_mask), dtype=np.bool_).transpose((2, 0, 1))
                #################################################################
                pred_masks.append(pred_mask)
                pred_labels.append(dat['pred_label'])
                pred_scores.append(dat['pred_score'])

        ap_result = eval_instance_segmentation_voc(pred_masks,
                                                   pred_labels,
                                                   pred_scores,
                                                   ins_gt_masks,
                                                   ins_gt_labels,
                                                   iou_thresh=0.25)

        print(ap_result)
        # os.system(f"rm -rf {val_dir}")

#     def calculate_gflops(model, input_size):
#         model_info = summary(model, input_size=input_size, verbose=0)
#         gflops = model_info.total_mult_adds / 1e9
#         return gflops

    torch.distributed.barrier()
#     input_size=(1, 3, target_size[0], target_size[1])
#     gflops = calculate_gflops(model, input_size)
#     print(f"GFLOPs: {gflops}")
#    print(img.size[0], img.size[1])
    model.train()

    return ap_result

# def validate(data_loader, args):
#     model.eval()
#
#     pred_seg_maps, pred_labels, pred_masks, pred_scores = [], [], [], []
#     val_dir = args.result_dir
#     if args.local_rank == 0:
#         os.makedirs(val_dir, exist_ok=True)
#
#     torch.distributed.barrier()
#     for img, cls_label, points, fname, tsize in tqdm(data_loader):
#         target_size = int(tsize[0]), int(tsize[1])
#
#         if args.val_flip:
#             img = torch.cat([img, img.flip(-1)], dim=0)
#
#         out = model(img.to(device), label=cls_label, target_shape=target_size)
#
#         if args.sup == 'point' and args.val_clean:
#             pred_seg, pred_label, pred_mask, pred_score, ann_pts = get_ins_map_with_point(out,
#                                                                                  cls_label,
#                                                                                  points,
#                                                                                  target_size,
#                                                                                  device,
#                                                                                  args)
#         else:
#             pred_seg, pred_label, pred_mask, pred_score = get_ins_map(out,
#                                                                       cls_label,
#                                                                       target_size,
#                                                                       device,
#                                                                       args)
#             ann_pts = []
#
#         with open(f'{val_dir}/{fname[0]}.pickle', 'wb') as f:
#             pickle.dump({
#                 "ann_pts": ann_pts,
#                 'pred_label': pred_label,  # attention: cls here start form 0, but in annotation start from 1
#                 'pred_mask': pred_mask,
#                 'pred_score': pred_score,
#             }, f)
#
#     torch.distributed.barrier()
#
#     ap_result = {"ap": None, "map": None}
#
#     if args.local_rank == 0 and args.dataset == 'voc':
#         pred_masks, pred_labels, pred_scores = [], [], []
#
#         for fname in ins_gt_ids:
#             with open(f'{val_dir}/{fname}.pickle', 'rb') as f:
#                 dat = pickle.load(f)
#                 pred_masks.append(dat['pred_mask'])
#                 pred_labels.append(dat['pred_label'])
#                 pred_scores.append(dat['pred_score'])
#
#         ap_result = eval_instance_segmentation_voc(pred_masks,
#                                                    pred_labels,
#                                                    pred_scores,
#                                                    ins_gt_masks,
#                                                    ins_gt_labels,
#                                                    iou_thresh=0.5)
#
#         print(ap_result)
#         # os.system(f"rm -rf {val_dir}")
#
#     torch.distributed.barrier()
#
#     model.train()
#
#     return ap_result


def aff_loss(pred_aff, label_aff):
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

if __name__ == '__main__':

    args = parse()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    torch.backends.cudnn.benchmark = True

    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)

    # Init dirstributed system
    torch.distributed.init_process_group(
        backend="nccl", rank=args.local_rank, world_size=torch.cuda.device_count()
    )
    args.world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{args.gpu}")

    if args.local_rank == 0:
        os.makedirs(args.save_folder, exist_ok=True)

    """ load model """
    model = model_factory(args)
    model = model.to(device)
#     target_size1 = (1, 3, 352, 512)  
#     # 计算并打印GFLOPs
#     gflops = calculate_gflops(model, target_size1, cls_label, device)
#     print(f"Model GFLOPs for representative input size {target_size1[2:]}: {gflops}")
    
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    # define loss function (criterion) and optimizer
    criterion = {"center": Weighted_MSELoss(),
                 "offset": Weighted_L1_Loss(),
                 "seg": DeepLabCE()
                 }

    ######################################
    # criterion['aff'] = nn.CrossEntropyLoss()
    ######################################

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_func("=> loading checkpoint '{}'".format(args.resume))
            ckpt = torch.load(args.resume, map_location='cpu')['model']
            model.load_state_dict(ckpt, strict=True)
        else:
            print_func("=> no checkpoint found at '{}'".format(args.resume))

    """ Get data loader """
    if args.dataset == 'voc':
        train_dataset = get_dataset_voc(args, mode='train', data_list="data/VOCdevkit/train_cls.txt")  # 10,582 images
        valid_dataset = get_dataset_voc(args, mode='val', data_list="data/VOCdevkit/train_cls.txt")
        print_func("VOC dataset used. Number of train set = %d | valid set = %d" % (len(train_dataset), len(valid_dataset)))
    elif args.dataset == 'coco':
        train_dataset = get_dataset_coco(args, mode='train')  # images
        valid_dataset = get_dataset_coco(args, mode='val')
        print_func("COCO dataset used. Number of train set = %d | valid set = %d" % (len(train_dataset), len(valid_dataset)))

    n_gpus = torch.cuda.device_count()
    batch_per_gpu = args.batch_size // n_gpus

    train_sampler = DistributedSampler(train_dataset, num_replicas=n_gpus, rank=args.local_rank)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_per_gpu,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              sampler=train_sampler,
                              drop_last=True)

    valid_sampler = DistributedSampler(valid_dataset, num_replicas=n_gpus, rank=args.local_rank)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=1,
                              num_workers=4,
                              pin_memory=True,
                              sampler=valid_sampler,
                              drop_last=False)

    if args.train_epoch != 0:
        args.train_iter = args.train_epoch * len(train_loader)

    lr_scheduler = WarmupPolyLR(
        optimizer,
        args.train_iter,
        warmup_iters=args.warm_iter,
        power=args.gamma,
    )

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)

    if args.val_freq != 0 and args.local_rank == 0:
        print("...Preparing GT dataset for evaluation")
        ins_dataset = VOCInstanceSegmentationDataset(split='train', data_dir=args.root_dir)

        ins_gt_ids = ins_dataset.ids
        ins_gt_masks = [ins_dataset.get_example_by_keys(i, (1,))[0] for i in range(len(ins_dataset))]
        ins_gt_labels = [ins_dataset.get_example_by_keys(i, (2,))[0] for i in range(len(ins_dataset))]

    torch.distributed.barrier()
    print_func("...Training Start \n")
    print_func(args)

    # validate()
    if args.mode == 'train':
        train()
    elif args.mode == 'validate':
        validate(valid_loader, args)
        valid_train_dataset = get_dataset_voc(args, mode='val', data_list="data/VOCdevkit/train_cls.txt")
        valid_train_loader = DataLoader(valid_train_dataset,
                                  batch_size=1,
                                  num_workers=4,
                                  pin_memory=True,
                                  sampler=valid_sampler,
                                  drop_last=False)
        validate(valid_train_loader, args)
