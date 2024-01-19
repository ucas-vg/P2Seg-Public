"""
BESTIE
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import numpy as np
import math
import cv2
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist


import time

MINIMUM_MASK_SIZE = 50
MAXIMUM_NUM_INST = 5
mask_pooling = torch.nn.MaxPool2d(kernel_size=4, stride=4)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synch(self, device):

        _sum = torch.tensor(self.sum).to(device)
        _count = torch.tensor(self.count).to(device)

        torch.distributed.reduce(_sum, dst=0)
        torch.distributed.reduce(_count, dst=0)

        if torch.distributed.get_rank() == 0:
            self.sum = _sum.item()
            self.count = _count.item()
            self.avg = self.sum / self.count

            
def gaussian(sigma=6):
    """
    2D Gaussian Kernel Generation.
    """
    size = 6 * sigma + 3
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    return g


def img_to_tensor(image):
    image = image.astype(np.float32)
    
    image /= 255.
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)
    
    return image.cuda()
    
    
def tensor_to_img(image):
    image = image.transpose(1, 2, 0)
    image *= [0.229, 0.224, 0.225]
    image += [0.485, 0.456, 0.406]
    image = np.uint8(image * 255)
    
    return image


def center_map_gen(center_map, x, y, label, sigma, g):
    """
    Center map generation. point to heatmap.
    Arguments:
        center_map: A Tensor of shape [C, H, W].
        x: A Int type value. x-coordinate for the center point.
        y: A Int type value. y-coordinate for the center point.
        label: A Int type value. class for the center point.
        sigma: A Int type value. sigma for 2D gaussian kernel.
        g: A numpy array. predefined 2D gaussian kernel to be encoded in the center_map.
        
    Returns:
        A numpy array of shape [C, H, W]. center map in which points are encoded in 2D gaussian kernel.
    """

    channel, height, width = center_map.shape

    # outside image boundary
    if x < 0 or y < 0 or x >= width or y >= height:
        return center_map
        
    # upper left
    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
    # bottom right
    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

    c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
    a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

    cc, dd = max(0, ul[0]), min(br[0], width)
    aa, bb = max(0, ul[1]), min(br[1], height)
    
    center_map[label, aa:bb, cc:dd] = np.maximum(
        center_map[label, aa:bb, cc:dd], g[a:b, c:d])
        
    return center_map


def extract_peak(heat, kernel=5, K=25, thresh=0.3):
    """
    Extract points from the center map. heatmap to point.
    Arguments:
        heat: A Tensor of shape [C, H, W]. center map.
        kernel: A Int type value. Kernel size for extract local maximum points.
        K: A maximum number of instances in the heat-map
        thresh: threshold for the heat-map
        
    Returns:
        A list. extracted class-wise points.
    """
    
    B, C, H, W = heat.size()
    
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)

    keep = (hmax == heat).float()
    
    peak = heat * keep
    
    topk_scores, topk_inds = torch.topk(peak.view(B, C, -1), K)

    topk_inds = topk_inds % (H * W)
    topk_ys   = (topk_inds / float(W)).int().float()
    topk_xs   = (topk_inds % W).int().float()
    
    topk_scores = topk_scores[0].float().detach().cpu().numpy()
    topk_ys = topk_ys[0].int().detach().cpu().numpy()
    topk_xs = topk_xs[0].int().detach().cpu().numpy()
    
    peaks = [[] for _ in range(C)]
    
    for cls in range(C):
        for conf, y, x in zip(topk_scores[cls], topk_ys[cls], topk_xs[cls]):
            if conf < thresh:
                break
            peaks[cls].append( (x, y) )
    
    return peaks


def pseudo_label_generation(sup, seg_map, point, cls_label, num_classes, sigma, g, args):
    """
    Pseudo-label generation (Semantic Knowledge Transfer).
    Arguments:
        sup: A String type. Weak supervision source (cls or point).
        seg_map: A numpy array [H, W]. weakly-supervised semantic segmentation map.
        point: A list. point label.
        cls_label: A numpy array. Image-level label.
        num_classes: A Int type value. number of classes.
        sigma: A Int type value. sigma for 2D gaussian kernel.
        g: A numpy array. predefined 2D gaussian kernel to be encoded in the center_map.
        
    Returns:
        center_map: A numpy array [C, H, W]. pseudo center map.
        offset_map: A numpy array [2, H, W]. pseudo offset map.
        weight_map: A numpy array [1, H, W]. weight map for the Instance-aware Guidance.
    """
    points = [[] for _ in range(num_classes)]
    
    for px, py, cls, conf in point:
        points[cls].append( (px, py) )

    H, W = seg_map.shape
    
    offset_map = np.zeros((2, H, W), dtype=np.float32)
    weight_map = np.zeros((1, H, W), dtype=np.float32)
    center_map = np.zeros((num_classes, H, W), dtype=np.float32)
        
    if sup == 'point':
        for cls in np.nonzero(cls_label)[0]:
            for cx, cy in points[cls]:
                center_map = center_map_gen(center_map, cx, cy, cls, sigma, g)
        
    y_coord = np.ones_like(seg_map, dtype=np.float32)
    x_coord = np.ones_like(seg_map, dtype=np.float32)
    y_coord = np.cumsum(y_coord, axis=0) - 1
    x_coord = np.cumsum(x_coord, axis=1) - 1
    # cc = 0
    masks = []
    for cls in np.nonzero(cls_label)[0]:
        mask = (seg_map == (cls+1)).astype(np.uint8)
        masks.append(mask)
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for k in range(1, nLabels):
            size = stats[k, cv2.CC_STAT_AREA]
            cx, cy = list(map(int, centroids[k]))

            if size < MINIMUM_MASK_SIZE:
                continue

            # check how many points are in a contour mask
            match_count = 0

            center_k = []

            for x, y in points[cls]:
                if labels[y, x] == k:
                    match_count += 1
                    center_k.append(np.array([y, x]))
                    
            # pseudo label generation
            if match_count == 1:  # accept : 1 contour - 1 point
                if sup == 'cls':
                    center_map = center_map_gen(center_map, cx, cy, cls, sigma, g)
                
                mask_index = np.where(labels == k)
                offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
                offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])

                weight_map[0][mask_index] = 1
                offset_map[offset_y_index] = cy - y_coord[mask_index]
                offset_map[offset_x_index] = cx - x_coord[mask_index]
            elif match_count > 1 and args.dis_group:
                mask_index = np.where(labels == k)
                offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
                offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
                weight_map[0][mask_index] = 1

                center_k = np.asarray(center_k)
                ind = np.asarray(mask_index).T
                dis = cdist(center_k, ind, 'euclid')
                cy = center_k[np.argmin(dis, axis=0)].T[0]
                cx = center_k[np.argmin(dis, axis=0)].T[1]

                offset_map[offset_y_index] = cy - y_coord[mask_index]
                offset_map[offset_x_index] = cx - x_coord[mask_index]
    # masks = np.stack(masks)
    # masks = max_pooling_mask(masks, (4, 4), 4)
    # aff_map = mask_to_aff(masks)

                # cc += 1
    # print('number of misses:', cc)
    masks = np.stack(masks)
    # masks = torch.from_numpy(masks).float().cuda()
    # masks = mask_pooling(masks)
    #
    #
    # # begin = time.time()
    #
    # aff_map = mask_to_aff(masks)
    # aff_map = aff_map.cpu()
    # end = time.time()
    # print('t: ', end - begin)

    return center_map, offset_map, weight_map, masks

def max_pooling_mask(masks, pooling_kernel_size, stride):
    c, h, w = masks.shape
    ph, pw = pooling_kernel_size
    outh = (h - ph) // stride + 1
    outw = (w - pw) // stride + 1
    out = np.zeros((c, outh, outw))

    # for i in range(c):
    #     for x in range(outh):
    #         for y in range(outw):
    #             out[i][x][y] = np.max(masks[i][x * stride:(x + 1) * stride, y * stride:(y + 1) * stride])
    return out.astype(int)


def refine_label_generation(seg_map, center_map, offset_map, label, gt_seg_map, args):
    """
    Refined-label generation (Self-Refinement) with image-level labels.
    Arguments:
        seg_map: A Tensor of shape [B, C+1, H, W]. output semantic segmentation map.
        center_map: A Tensor of shape [B, C, H, W]. output center map.
        offset_map: A Tensor of shape [B, 2, H, W]. output offset map.
        label: A Tensor of shape [B, C]. one-hot image-level label.
        gt_seg_map: A Tensor of shape [B, H, W]. ground-truth (weakly-supervised) semantic segmentation map.
        args: arguments

    Returns:
        dictionary type:
            refined_center_map: A Tensor of shape [B, C, H, W]. refined center map.
            refined_offset_map: A Tensor of shape [B, 2, H, W]. refined offset map.
            refined_weight_map: A numpy array [1, H, W]. refined weight map for the Instance-aware Guidance.
    """
    device = seg_map.device
    B, C, H, W = center_map.shape
    
    prob_map = F.softmax(seg_map, dim=1)        # [B, C+1, H, W]
    prob_map[:, 1:, :, :] *= label[:, :, None, None] # cleaning using image-level label
    seg_map = gt_seg_map
    
    refined_offset_map = torch.zeros((B, 2, H, W), dtype=torch.float32).to(device)
    refined_weight_map = torch.zeros((B, 1, H, W), dtype=torch.float32).to(device)
    refined_center_map = np.zeros((B, C, H, W), dtype=np.float32)
    
    y_coord = torch.ones((H, W), dtype=torch.float32).to(device)
    x_coord = torch.ones((H, W), dtype=torch.float32).to(device)
    y_coord = torch.cumsum(y_coord, dim=0) - 1
    x_coord = torch.cumsum(x_coord, dim=1) - 1

    label = label.cpu().numpy()
    g = gaussian(args.sigma)
    
    for b in range(B):
        _seg_map = seg_map[b]        # [C+1, H, W]
        _center_map = center_map[b]  # [C, H, W]
        _offset_map = offset_map[b]  # [2, H, W]
        _valid_cls = np.nonzero(label[b])[0]
        
        for _cls in _valid_cls:
            __center_map = _center_map[_cls]      # [H, W]
            __foreground = (_seg_map == (_cls+1)).bool() # [H, W]
            
            __fg_mask = __foreground.cpu().numpy().astype(np.uint8)
            n_contours, contours, stats, centroids = cv2.connectedComponentsWithStats(__fg_mask, connectivity=8)
            
            for k in range(1, n_contours):
                size = stats[k, cv2.CC_STAT_AREA]
                cx, cy = list(map(int, centroids[k]))
            
                if size < MINIMUM_MASK_SIZE:
                    continue
                
                contour_mask = (contours == k)
                contour_mask = torch.from_numpy(contour_mask).to(device) # [H, W]

                __c_center_map = __center_map * contour_mask

                __ins_seg = get_instance_segmentation(contour_mask[None, ...], 
                                                      __c_center_map[None, None, ...], 
                                                      _offset_map[None, ...],
                                                      threshold=args.refine_thresh, 
                                                      nms_kernel=args.kernel,
                                                      ignore=True,
                                                      beta=args.beta,
                                                     )

                __ins_seg = __ins_seg.squeeze(0)
                n_ins = __ins_seg.max()

                # too many centers in single contour
                if n_ins > MAXIMUM_NUM_INST:
                    continue
                
                for i in range(1, n_ins+1):

                    mask = (__ins_seg == i)

                    if mask.sum() > 0:
                        index = torch.where(mask)

                        pmax = __c_center_map[index].argmax()
                        seg_score = prob_map[b, _cls+1][index].mean().item()

                        py, px = index[0][pmax].item(), index[1][pmax].item()
                        
                        center_score = __c_center_map[py, px].item()

                        """  using seg_mask as ins_mask, it's center is contour's center"""
                        if center_score < args.refine_thresh: # ins mask <- seg mask
                            py, px = cy, cx # choice 2
                            conf = seg_score
                        else:
                            conf = center_score * seg_score
                            
                        conf = max(0, min(conf, 1))

                        refined_center_map[b] = center_map_gen(
                            refined_center_map[b], 
                            px, py, _cls, 
                            args.sigma, g, 
                        )

                        offset_y_index = (torch.zeros_like(index[0]), index[0], index[1])
                        offset_x_index = (torch.ones_like(index[0]), index[0], index[1])

                        refined_weight_map[b, 0][index] = conf
                        refined_offset_map[b][offset_y_index] = py - y_coord[index]
                        refined_offset_map[b][offset_x_index] = px - x_coord[index]

    refined_center_map = torch.from_numpy(refined_center_map).to(device)
    
    return {
        'center' : refined_center_map,
        'offset' : refined_offset_map, 
        'weight' : refined_weight_map, 
    }
    

def refine_label_generation_with_point(seg_map, gt_point_cls, offset_map, label, gt_seg_map, args):
    """
    Refined-label generation (Self-Refinement) with point labels.
    Arguments:
        seg_map: A Tensor of shape [B, C+1, H, W]. output semantic segmentation map.
        gt_point_cls : A Tensor of shape [B, C, MAX_NUM_POINTS, 2)], ground-truth class-wise point label.
        offset_map: A Tensor of shape [B, 2, H, W]. output offset map.
        label: A Tensor of shape [B, C]. one-hot image-level label.
        gt_seg_map: A Tensor of shape [B, H, W]. ground-truth (weakly-supervised) semantic segmentation map.
        args: arguments

    Returns:
        dictionary type:
            refined_offset_map: A Tensor of shape [B, 2, H, W]. refined offset map.
            refined_weight_map: A numpy array [1, H, W]. refined weight map for the Instance-aware Guidance.
    """
    device = seg_map.device
    B, C, H, W = seg_map.shape
    
    seg_prob = F.softmax(seg_map, dim=1)        # [B, C+1, H, W]
    seg_prob[:, 1:, :, :] *= label[:, :, None, None]
    seg_map = gt_seg_map
    
    pseudo_offset_map = torch.zeros((B, 2, H, W), dtype=torch.float32).to(device)
    pseudo_weight_map = torch.zeros((B, 1, H, W), dtype=torch.float32).to(device)
    
    y_coord = torch.ones((H, W), dtype=torch.float32).to(device)
    x_coord = torch.ones((H, W), dtype=torch.float32).to(device)
    y_coord = torch.cumsum(y_coord, dim=0) - 1
    x_coord = torch.cumsum(x_coord, dim=1) - 1

    label = label.cpu().numpy()

    pseudo_masks = torch.zeros((B, H, W), dtype=torch.float32).to(device)
    
    for b in range(B):
        _seg_map = seg_map[b]        # [C+1, H, W]
        _gt_point_cls = gt_point_cls[b]
        _offset_map = offset_map[b]  # [2, H, W]
        _valid_cls = np.nonzero(label[b])[0]
        
        for cls in _valid_cls:
            _foreground = (_seg_map == (cls+1)).bool()   # [H, W]
            
            _gt_point = [ (gt_y, gt_x) for gt_y, gt_x in _gt_point_cls[cls] if gt_y !=0 and gt_x != 0]
            _gt_point = np.int32(_gt_point)

            if _gt_point.shape[0] <= 0:
                continue
                
            _gt_point = torch.from_numpy(_gt_point).long().to(device)
            
            ins_seg = group_pixels(_gt_point, _offset_map.unsqueeze(0))
            ing_seg = (_foreground * ins_seg).squeeze(0).long()

            n_ins = ing_seg.max()

            for i in range(1, n_ins+1):
                mask = (ing_seg == i) # [H, W]

                if mask.sum() > 0:
                    index = torch.where(mask)

                    cy, cx = _gt_point[i-1]
                    
                    offset_y_index = (torch.zeros_like(index[0]), index[0], index[1])
                    offset_x_index = (torch.ones_like(index[0]), index[0], index[1])

                    pseudo_weight_map[b, 0][index] = 1
                    pseudo_offset_map[b][offset_y_index] = cy - y_coord[index]
                    pseudo_offset_map[b][offset_x_index] = cx - x_coord[index]

                    pseudo_masks[b] = mask

    return {'offset' : pseudo_offset_map, 
            'weight' : pseudo_weight_map,
            'masks'  : pseudo_masks
           }

def mask_to_aff(mask):
    """
    mask: array(b, h, w)
    """
    begin = time.time()
    N = mask.shape[0]
    mask_flatten = mask.reshape(N, -1) #(b, -1)
    mask_flatten = mask_flatten.unsqueeze(-1)

    out = torch.matmul(mask_flatten, mask_flatten.permute(0, 2, 1))

    out = torch.sum(out, dim=0)

    out = torch.where(out > 0, 1, 0)

    print('t: ', time.time() - begin)
    # out = out.cpu()

    return out
    
def find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=5, top_k=None):
    """
    # This implementation is from https://github.com/bowenc0221/panoptic-deeplab.
    Find the center points from the center heatmap.
    Arguments:
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
    Returns:
        A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    # thresholding, setting values below threshold to -1
    ctr_hmp = F.threshold(ctr_hmp, threshold, -1)

    # NMS
    nms_padding = (nms_kernel - 1) // 2
    ctr_hmp_max_pooled = F.max_pool2d(ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding)
    ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1

    # squeeze first two dimensions
    ctr_hmp = ctr_hmp.squeeze()
    assert len(ctr_hmp.size()) == 2, 'Something is wrong with center heatmap dimension.'

    # find non-zero elements
    ctr_all = torch.nonzero(ctr_hmp > 0, as_tuple=False)
    
    if top_k is None:
        return ctr_all
    elif ctr_all.size(0) < top_k:
        return ctr_all
    else:
        # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(ctr_all), top_k)
        return torch.nonzero(ctr_hmp > top_k_scores[-1], as_tuple=False)
    
    
def group_pixels(ctr, offsets):
    """
    # This implementation is from https://github.com/bowenc0221/panoptic-deeplab.
    Gives each pixel in the image an instance id.
    Arguments:
        ctr: A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    """
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    offsets = offsets.squeeze(0)
    height, width = offsets.size()[1:]

    # generates a coordinate map, where each location is the coordinate of that loc
    y_coord = torch.arange(height, dtype=offsets.dtype, device=offsets.device).repeat(1, width, 1).transpose(1, 2)
    x_coord = torch.arange(width, dtype=offsets.dtype, device=offsets.device).repeat(1, height, 1)
    coord = torch.cat((y_coord, x_coord), dim=0)

    ctr_loc = coord + offsets
    ctr_loc = ctr_loc.reshape((2, height * width)).transpose(1, 0)

    # ctr: [K, 2] -> [K, 1, 2]
    # ctr_loc = [H*W, 2] -> [1, H*W, 2]
    ctr = ctr.unsqueeze(1)
    ctr_loc = ctr_loc.unsqueeze(0)

    # distance: [K, H*W]
    distance = torch.norm(ctr - ctr_loc, dim=-1)

    # finds center with minimum distance at each location, offset by 1, to reserve id=0 for stuff
    # numober of instance = K
    instance_id = torch.argmin(distance, dim=0).reshape((1, height, width)) + 1
    
    return instance_id


def get_instance_segmentation(fg, ctr_hmp, offsets, threshold=0.1, nms_kernel=3, top_k=None, ignore=True, beta=5):
    """
    # This implementation is from https://github.com/bowenc0221/panoptic-deeplab.
    Post-processing for instance segmentation, gets class agnostic instance id map.
    
    Arguments:
        fg: A Tensor of shape [B, H, W], foreground map.
        ctr_hmp: A Tensor of shape [B, 1, H, W] of raw center heatmap output, where B is the batch size,
            for consistent, we only support B=1.
        offsets: A Tensor of shape [B, 2, H, W] of raw offset output, where B is the batch size,
            for consistent, we only support B=1. The order of second dim is (offset_y, offset_x).
        thing_list: A List of thing class id.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
        A Tensor of shape [1, K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    
    ctr = find_instance_center(ctr_hmp, threshold=threshold, nms_kernel=nms_kernel, top_k=top_k)
    
    if beta > 0: # center clustring
        try:
            ctr_cluster = cluster_peaks(offsets[0].cpu().numpy(), fg[0].cpu().numpy(), beta=beta)
            ctr_cluster = np.int32([[cy, cx] for cy, cx in ctr_cluster if ctr_hmp[0, 0, cy, cx] > 0.05])
            ctr_cluster = torch.from_numpy(ctr_cluster).to(ctr.device).long()

            new_ctr = ctr.clone()

            # ctr & ctr_cluster merge
            if ctr_cluster.size(0) > 0:

                if ctr.size(0) == 0:
                    new_ctr = ctr_cluster

                    """ mark as new peak """
                    for cy, cx in ctr_cluster:
                        ctr_hmp[0, 0, cy, cx] = 1.0

                else:
                    # merge points without overlap
                    for c_cluster in ctr_cluster:
                        c_min_dist = torch.norm(ctr.float() - c_cluster.float(), dim=-1).min()

                        if c_min_dist > 100:
                            new_ctr = torch.cat([new_ctr, c_cluster.unsqueeze(0)], dim=0)
                            ctr_hmp[0, 0, c_cluster[0], c_cluster[1]] = 1.0 # mark as new peak
        except:
            new_ctr = ctr
    else:
        new_ctr = ctr
        
    if new_ctr.size(0) == 0: # no peak & no cluster
        if ignore:
            return torch.zeros_like(fg).long() #, new_ctr.unsqueeze(0)
        else:
            return fg.long() #, new_ctr.unsqueeze(0)
        
    ins_seg = group_pixels(new_ctr, offsets)
    
    return (fg * ins_seg).long() #, new_ctr.unsqueeze(0)


def cluster_peaks(offset_map, fg, thresh=2.5, beta=5):
    """
    Center clustering (offset map -> clustered point).
    
    Arguments:
        offset_map: A numpy array of shape [2, H, W]. output offset map.
        fg: A numpy array of shape [H, W]. one-hot image-level label.
        thresh: A float type value. threshold for center grouping.
        beta: A int type value. epsilon value for the clustering
    Returns:
        clustered point.
    """
    magnitude = np.sqrt(offset_map[1] ** 2 + offset_map[0] ** 2)
    
    height, width = magnitude.shape
    
    weak_dp_region = (magnitude < thresh)
    weak_dp_region *= fg
    
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(weak_dp_region.astype(np.uint8), connectivity=4)
    
    """ centroid is [x, y] order """
    peaks = [ centroids[k][::-1] for k in range(nLabels) if 21-beta < stats[k, cv2.CC_STAT_AREA] < 21+beta  ]
    
    return np.int32(peaks)


def get_ins_map(out, cls_label, target_size, device, args):
    """
    post-processing (output -> instance map).
    
    Arguments:
        out: A Dict type for network outputs.
            out['seg']: output semantic segmentation logits
            out['center']: output center map
            out['offset']: output offset map
        cls_label: A Tensor of shape [B, C]. one-hot image-level label.
        target_size: A list for target size (H, W)
        device: output device.
        args: arguments

    Returns:
        seg_map: A numpy array of shape [H, W]. output semantic segmentation map.
        pred_label: A numpy array of shape [H, W]. class-label for output instance mask.
        pred_mask: A numpy array of shape [H, W]. pixel-wise mask for output instance mask.
        pred_score: A numpy array of shape [H, W]. confidence score for output instance mask.
    """
    pred_seg, pred_label, pred_mask, pred_score = [], [], [], []

    seg_prob = torch.softmax(out['seg'].detach(), 1) # B, C+1, H, W
    center_map = out['center'].detach()
    offset_map = out['offset'][0].detach()
    
    if args.val_flip:
        seg_prob = (seg_prob[0] + seg_prob[1].flip(-1)) / 2. # [C+1, H, W]
        center_map = (center_map[0] + center_map[1].flip(-1)) / 2. # [C, H, W]
    else:
        seg_prob = seg_prob[0]
        center_map = center_map[0]

    out_size = seg_prob.shape[1:]
    offset_map[0, :, :] = offset_map[0, :, :] * (target_size[0] / out_size[0])
    offset_map[1, :, :] = offset_map[1, :, :] * (target_size[1] / out_size[1])

    if args.val_clean:
        seg_prob[1:, :, :] *= cls_label[0, :, None, None].to(device)

    seg_map = torch.argmax(seg_prob, 0)
    valid_cls = torch.unique(seg_map).cpu().numpy() - 1 # -1 for removing bg-class
    
    for cls in valid_cls:
        if cls < 0: 
            continue
        center_map_cls = center_map[cls]         # [H, W]
        fg_cls = (seg_map == (cls+1)).bool()   # [H, W]

        fg_cls = fg_cls.cpu().numpy().astype(np.uint8)
        n_contours, contours, stats, _ = cv2.connectedComponentsWithStats(fg_cls, connectivity=8)

        for k in range(1, n_contours):
            size = stats[k, cv2.CC_STAT_AREA]

            if size < MINIMUM_MASK_SIZE:
                continue

            contour_mask = (contours == k)
            contour_mask = torch.from_numpy(contour_mask).to(device) # [H, W]

            center_map_cls_roi = center_map_cls * contour_mask # get roi in center_map

            ins_map = get_instance_segmentation(
                contour_mask[None, ...], 
                center_map_cls_roi[None, None, ...], 
                offset_map[None, ...],
                threshold=args.val_thresh, 
                nms_kernel=args.val_kernel,
                beta=args.beta,
                ignore=args.val_ignore
            )

            # ins_map : [1, H, W]
            # seg_prob : [21, H, W]
            ins_map = ins_map.squeeze(0)
            n_ins = ins_map.max()

            for id in range(1, n_ins+1):
                mask = (ins_map == id) # [H, W]

                if mask.sum() > 0:
                    index = torch.where(mask)

                    center_idx = center_map_cls_roi[index].argmax()
                    seg_score = seg_prob[cls+1][index].mean().item()

                    cy, cx = index[0][center_idx], index[1][center_idx]
                    center_score = center_map_cls_roi[cy, cx].item()

                    if center_score >= 1: # clustered center conf = seg_score
                        center_score = seg_score

                    pred_label.append(cls)
                    pred_mask.append(mask.cpu().numpy())
                    pred_score.append(center_score * seg_score)

    if len(pred_label) == 0:
        pred_label.append(0)
        pred_mask.append(np.zeros(target_size, dtype=np.bool))
        pred_score.append(0)

    pred_label = np.stack(pred_label, 0)
    pred_mask = np.stack(pred_mask, 0)
    pred_score = np.stack(pred_score, 0)
    
    return seg_map.cpu().numpy(), pred_label, pred_mask, pred_score


def get_ins_map_with_point(out, cls_label, points, target_size, device, args):
    """
    post-processing (output -> instance map) using ground-truth point labels.
    
    Arguments:
        out: A Dict type for network outputs.
            out['seg']: output semantic segmentation logits
            out['center']: output center map
            out['offset']: output offset map
        cls_label: A Tensor of shape [B, C]. one-hot image-level label.
        points: ground-truth point labels.
        target_size: A list for target size (H, W)
        device: output device.
        args: arguments

    Returns:
        seg_map: A numpy array of shape [H, W]. output semantic segmentation map.
        pred_label: A numpy array of shape [H, W]. class-label for output instance mask.
        pred_mask: A numpy array of shape [H, W]. pixel-wise mask for output instance mask.
        pred_score: A numpy array of shape [H, W]. confidence score for output instance mask.
    """
    pred_seg, pred_label, pred_mask, pred_score, pred_pts = [], [], [], [], []

    seg_prob = torch.softmax(out['seg'].detach(), 1)  # B, C+1, H, W
    center_map = out['center'].detach()
    offset_map = out['offset'][0].detach()

    if args.val_flip:
        seg_prob = (seg_prob[0] + seg_prob[1].flip(-1)) / 2.  # [C+1, H, W]
        center_map = (center_map[0] + center_map[1].flip(-1)) / 2.  # [C, H, W]
    else:
        seg_prob = seg_prob[0]
        center_map = center_map[0]

    out_size = seg_prob.shape[1:]
    offset_map[0, :, :] = offset_map[0, :, :] * (target_size[0] / out_size[0])
    offset_map[1, :, :] = offset_map[1, :, :] * (target_size[1] / out_size[1])
    seg_prob[1:, :, :] *= cls_label[0, :, None, None].to(device)

    seg_map = torch.argmax(seg_prob, 0)
    valid_cls = torch.unique(seg_map).cpu().numpy() - 1 #  -1 for removing bg-class

    for cls in valid_cls:
        if cls < 0:
            continue
        center_map_cls = center_map[cls]         # [H, W]
        fg_cls = (seg_map == (cls+1)).bool()     # [H, W]

        points_cls = np.int32(points[cls])

        if len(points[cls]) > 0:
            points_cls = torch.tensor(points_cls, dtype=torch.long, device=device)
            ins_map = group_pixels(points_cls, offset_map.unsqueeze(0))
            # ng_ins = ins_map.max()
            ins_map = (fg_cls * ins_map).squeeze(0).long()
            n_ins = ins_map.max()

            for i, p in enumerate(points_cls.cpu().numpy()):
                mask = (ins_map == i + 1)
                if mask.sum() > 0:  # have mask for point i
                    index = torch.where(mask)
                    seg_score = seg_prob[cls + 1][index].mean().item()
                else:  # no mask for point i
                    # mask = 0
                    seg_score = 0
                pred_label.append(cls)               # cls start form 0, but annotation start from 1
                pred_mask.append(mask.cpu().numpy())
                pred_score.append(seg_score)
                pred_pts.append(p)

            # pred_pts.append([-1, -1])
            # for id in range(1, n_ins + 1):
            #     mask = (ins_map == id)  # [H, W]
            #
            #     if mask.sum() > 0:
            #         index = torch.where(mask)
            #         seg_score = seg_prob[cls + 1][index].mean().item()
            #
            #         pred_label.append(cls)
            #         pred_mask.append(mask.cpu().numpy())
            #         pred_score.append(seg_score)
    if len(pred_label) == 0:
        pred_label.append(0)
        pred_mask.append(np.zeros(target_size, dtype=np.bool))
        pred_score.append(0)
        pred_pts.append([-1, -1])

    pred_label = np.stack(pred_label, 0)
    pred_mask = np.stack(pred_mask, 0)
    pred_score = np.stack(pred_score, 0)
    pred_pts = np.stack(pred_pts, 0)

    return seg_map.cpu().numpy(), pred_label, pred_mask, pred_score, pred_pts

def get_ins_map_with_point_forallpoint(out, cls_label, points, points_c, target_size, device, args):
    """
    post-processing (output -> instance map) using ground-truth point labels.

    Arguments:
        out: A Dict type for network outputs.
            out['seg']: output semantic segmentation logits
            out['center']: output center map
            out['offset']: output offset map
        cls_label: A Tensor of shape [B, C]. one-hot image-level label.
        points: ground-truth point labels.
        target_size: A list for target size (H, W)
        device: output device.
        args: arguments

    Returns:
        seg_map: A numpy array of shape [H, W]. output semantic segmentation map.
        pred_label: A numpy array of shape [H, W]. class-label for output instance mask.
        pred_mask: A numpy array of shape [H, W]. pixel-wise mask for output instance mask.
        pred_score: A numpy array of shape [H, W]. confidence score for output instance mask.
    """
    pred_seg, pred_label, pred_mask, pred_score, pred_pts = [], [], [], [], []

    seg_prob = torch.softmax(out['seg'].detach(), 1)  # B, C+1, H, W
    center_map = out['center'].detach()
    offset_map = out['offset'][0].detach()

    if args.val_flip:
        seg_prob = (seg_prob[0] + seg_prob[1].flip(-1)) / 2.  # [C+1, H, W]
        center_map = (center_map[0] + center_map[1].flip(-1)) / 2.  # [C, H, W]
    else:
        seg_prob = seg_prob[0]
        center_map = center_map[0]

    out_size = seg_prob.shape[1:]
    offset_map[0, :, :] = offset_map[0, :, :] * (target_size[0] / out_size[0])
    offset_map[1, :, :] = offset_map[1, :, :] * (target_size[1] / out_size[1])
    seg_prob[1:, :, :] *= cls_label[0, :, None, None].to(device)

    seg_map = torch.argmax(seg_prob, 0)
    valid_cls = torch.unique(seg_map).cpu().numpy() - 1  # -1 for removing bg-class

    for cls in valid_cls:
        if cls < 0:
            continue
        center_map_cls = center_map[cls]  # [H, W]
        fg_cls = (seg_map == (cls + 1)).bool()  # [H, W]

        points_cls = np.int32(points_c[cls])

        if len(points_c[cls]) > 0:
            points_cls = torch.tensor(points_cls, dtype=torch.long, device=device)
            ins_map = group_pixels(points_cls, offset_map.unsqueeze(0))
            # ng_ins = ins_map.max()
            ins_map = (fg_cls * ins_map).squeeze(0).long()
            n_ins = ins_map.max()

            for i, p in enumerate(points_cls.cpu().numpy()):
                mask = (ins_map == i + 1)
                if mask.sum() > 0:  # have mask for point i
                    index = torch.where(mask)
                    seg_score = seg_prob[cls + 1][index].mean().item()
                else:  # no mask for point i
                    # mask = 0
                    seg_score = 0
                pred_label.append(cls)  # cls start form 0, but annotation start from 1
                pred_mask.append(mask.cpu().numpy())
                pred_score.append(seg_score)
                pred_pts.append(p)

            # pred_pts.append([-1, -1])
            # for id in range(1, n_ins + 1):
            #     mask = (ins_map == id)  # [H, W]
            #
            #     if mask.sum() > 0:
            #         index = torch.where(mask)
            #         seg_score = seg_prob[cls + 1][index].mean().item()
            #
            #         pred_label.append(cls)
            #         pred_mask.append(mask.cpu().numpy())
            #         pred_score.append(seg_score)
#     points_t = []
#     for ps in points:
#         points_t.append(np.array([ps[0], ps[1]], dtype=np.int64))

#     if len(pred_pts) < len(points):
#         for p in range(len(points)):
#             print(pred_pts)
#             print(points_t[p])            
#             if points_t[p] not in pred_pts:
#                 pred_label.append(points[p][2])  # cls start form 0, but annotation start from 1
#                 m = np.zeros(target_size, dtype=np.bool)
#                 boxsize = 16
#                 # m[max(points_t[p][0]-boxsize, 0):min(points_t[p][0]+boxsize, target_size[0],points_t[p][1]-boxsize:points_t[p][1]+boxsize] = True
#                 m[max(points_t[p][0]-boxsize, 0):min(points_t[p][0]+boxsize, target_size[0]), \
#                                 max(points_t[p][1]-boxsize, 0):min(points_t[p][1]+boxsize, target_size[1])] = True
#                 pred_mask.append(m)
#                 pred_score.append(0.5)
#                 pred_pts.append(points_t[p])
     
    points_t_x = []
    points_t_y = []
    for ps in points:
        points_t_x.append(ps[0])
        points_t_y.append(ps[1])
    boxsize = 32
    if pred_pts:
        p_pts = np.stack(pred_pts, 0)
        if len(pred_pts) < len(points):
            for p in range(len(points)):
                if int(points_t_x[p]) not in p_pts[:, 1] or int(points_t_y[p]) not in p_pts[:, 0]:
                    # if points_t[p] not in pred_pts:
                    pred_label.append(int(points[p][2]))  # cls start form 0, but annotation start from 1
                    m = np.zeros(target_size, dtype=np.bool)
                    # m[max(points_t[p][0]-boxsize, 0):min(points_t[p][0]+boxsize, target_size[0],points_t[p][1]-boxsize:points_t[p][1]+boxsize] = True
                    m[int(max(points_t_y[p] - boxsize, 0)):int(min(points_t_y[p] + boxsize, target_size[0])), \
                    int(max(points_t_x[p] - boxsize, 0)):int(min(points_t_x[p] + boxsize, target_size[1]))] = True
                    assert m.sum() > 0, f'{m.sum(), points_t_y[p], points_t_x[p]}'
                    pred_mask.append(m)
                    pred_score.append(0.5)
                    pred_pts.append(np.array([points_t_y[p], points_t_x[p]], dtype=np.int64))
#                     print(1)
    else:
#         print(0, len(pred_pts), len(points))
        for p in range(len(points)):
            # if points_t[p] not in pred_pts:
            # if points_t[p] not in pred_pts:
            pred_label.append(int(points[p][2]))  # cls start form 0, but annotation start from 1
            m = np.zeros(target_size, dtype=np.bool)
            # m[max(points_t[p][0]-boxsize, 0):min(points_t[p][0]+boxsize, target_size[0],points_t[p][1]-boxsize:points_t[p][1]+boxsize] = True
            m[int(max(points_t_y[p] - boxsize, 0)):int(min(points_t_y[p] + boxsize, target_size[0])), \
            int(max(points_t_x[p] - boxsize, 0)):int(min(points_t_x[p] + boxsize, target_size[1]))] = True
            assert m.sum() > 0, f'{m.sum(), points_t_y[p], points_t_x[p]}'
            pred_mask.append(m)
            pred_score.append(0.5)
            pred_pts.append(np.array([points_t_y[p], points_t_x[p]], dtype=np.int64))
#         print(0, len(pred_pts), len(points))
    assert len(pred_pts) == len(points), f'{len(pred_pts), len(points)}'
    # if len(pred_label) == 0:
    #     pred_label.append(0)
    #     pred_mask.append(np.zeros(target_size, dtype=np.bool))
    #     pred_score.append(0)
    #     pred_pts.append([-1, -1])

    pred_label = np.stack(pred_label, 0)
    pred_mask = np.stack(pred_mask, 0)
    pred_score = np.stack(pred_score, 0)
    pred_pts = np.stack(pred_pts, 0)

    return seg_map.cpu().numpy(), pred_label, pred_mask, pred_score, pred_pts


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def parse():
    parser = argparse.ArgumentParser(description='BESTIE pytorch implementation')
    parser.add_argument("--root_dir", type=str, default='data/VOC2012', help='Root dir for the project')
    parser.add_argument('--sup', type=str, help='supervision source', choices=["cls", "point"], default='point')
    parser.add_argument("--dataset", type=str, default='voc', choices=["voc", "coco"])
    parser.add_argument("--backbone", type=str, default='hrnet48',
                        choices=["resnet50", "resnet101", "hrnet34", "hrnet48"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=416)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--train_iter", type=int, default=50000)
    parser.add_argument("--warm_iter", type=int, default=2000, help='warm-up iterations')
    parser.add_argument("--train_epoch", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
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
    parser.add_argument("--local_rank", type=int, default=1)

    return parser.parse_args()


def print_func(string):
    if torch.distributed.get_rank() == 0:
        print(string)


if __name__ == '__main__':
    import argparse
    from LoadData import get_dataset
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from chainercv.evaluations import eval_instance_segmentation_voc
    from chainercv.datasets import VOCInstanceSegmentationDataset
    import random
    import os
    from tqdm import tqdm
    import pickle
    import numpy as np

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

    train_dataset = get_dataset(args, mode='train')
    valid_dataset = get_dataset(args, mode='test_self')
    print_func("number of train set = %d | valid set = %d" % (len(train_dataset), len(valid_dataset)))

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
                              num_workers=0,
                              pin_memory=True,
                              sampler=valid_sampler,
                              drop_last=False)
    if args.val_freq != 0 and args.local_rank == 0:
        print("...Preparing GT dataset for evaluation")
        ins_dataset = VOCInstanceSegmentationDataset(split='val', data_dir=args.root_dir)

        ins_gt_ids = ins_dataset.ids
        ins_gt_masks = [ins_dataset.get_example_by_keys(i, (1,))[0] for i in range(len(ins_dataset))]
        ins_gt_labels = [ins_dataset.get_example_by_keys(i, (2,))[0] for i in range(len(ins_dataset))]
    torch.distributed.barrier()

    val_dir = "val_temp_dir"
    device = torch.device(f"cuda:{args.gpu}")
    if args.local_rank == 0:
        os.makedirs(val_dir, exist_ok=True)

    torch.distributed.barrier()

    for img, label, seg_map, center_map, offset_map, points, fname, tsize in tqdm(valid_loader):
        target_size = int(img.size[0]), int(img.size[1])
        out = {'center':center_map, 'offset':offset_map, 'seg':seg_map}
        pred_seg, pred_label, pred_mask, pred_score = get_ins_map_with_point(out, label, points, target_size, device,
                                                                             args)
        with open(f'{val_dir}/{fname[0]}.pickle', 'wb') as f:
            pickle.dump({
                'pred_label': pred_label,
                'pred_mask': pred_mask,
                'pred_score': pred_score,
            }, f)
    torch.distributed.barrier()

    ap_result = {"ap": None, "map": None}

    if args.local_rank == 0:
        pred_masks, pred_labels, pred_scores = [], [], []

        for fname in ins_gt_ids:
            with open(f'{val_dir}/{fname}.pickle', 'rb') as f:
                dat = pickle.load(f)
                pred_masks.append(dat['pred_mask'])
                pred_labels.append(dat['pred_label'])
                pred_scores.append(dat['pred_score'])

        ap_result = eval_instance_segmentation_voc(pred_masks,
                                                   pred_labels,
                                                   pred_scores,
                                                   ins_gt_masks,
                                                   ins_gt_labels,
                                                   iou_thresh=0.5)

        print(ap_result)
        os.system(f"rm -rf {val_dir}")

    torch.distributed.barrier()