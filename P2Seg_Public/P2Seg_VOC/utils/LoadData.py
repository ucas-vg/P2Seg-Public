"""
BESTIE
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import os
from PIL import Image
import numpy as np

import torch
import torchvision.transforms.functional as TF
from .transforms import transforms as T


from torch.utils.data import Dataset
from .utils import gaussian, pseudo_label_generation

from pycocotools.coco import COCO

def get_dataset_voc(args, mode, data_list):  # add data_list by hui
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
        
    if mode == 'train':
        # data_list = "data/VOCdevkit/train_cls.txt" # 10,582 images
        
        crop_size = int(args.crop_size)
        input_size = crop_size + 64
    
        min_resize_value = input_size
        max_resize_value = input_size
        resize_factor = 32

        min_scale = 0.7
        max_scale = 1.3
        scale_step_size=0.1
        crop_h, crop_w = crop_size, crop_size

        pad_value = tuple([int(v * 255) for v in mean_vals])
        ignore_label = (0, 0, 0)

        transform = T.Compose(
            [
                T.PhotometricDistort(),
                T.Resize(min_resize_value, 
                         max_resize_value, 
                         resize_factor),
                T.RandomScale(
                    min_scale,
                    max_scale,
                    scale_step_size
                ),
                T.RandomCrop(
                    crop_h,
                    crop_w,
                    pad_value,
                    ignore_label,
                    random_pad=True
                ),
                T.ToTensor(),
                T.RandomHorizontalFlip(),
                T.Normalize(
                    mean_vals,
                    std_vals
                )
            ]
        )
        
        dataset = VOCDataset(data_list, 
                             root_dir=args.root_dir, 
                             num_classes=args.num_classes, 
                             transform=transform, 
                             sup=args.sup,
                             sigma=args.sigma, 
                             point_thresh=args.pseudo_thresh,
                             args=args)
    elif mode == 'test_self':
        # data_list = "data/VOCdevkit/val_cls.txt"
        dataset = VOCTestSelfDataset(data_list,
                                 root_dir=args.root_dir,
                                 num_classes=args.num_classes,
                                     args=args)
    else:
        #data_list = "data/train_labeled_cls.txt"
        # data_list = "data/VOCdevkit/val_cls.txt"
        if args.all_point:
            dataset = VOCTestAllPointDataset(data_list, 
                                 root_dir=args.root_dir,
                                 num_classes=args.num_classes,
                                 args=args)
        else:
            dataset = VOCTestDataset(data_list, 
                                 root_dir=args.root_dir,
                                 num_classes=args.num_classes,
                                 args=args)
    return dataset


def get_dataset_coco(args, mode):
    mean_vals = [0.471, 0.448, 0.408]
    std_vals = [0.234, 0.239, 0.242]
    if mode == 'train':
        crop_size = int(args.crop_size)
        input_size = crop_size + 64

        min_resize_value = input_size
        max_resize_value = input_size
        resize_factor = 32

        min_scale = 0.7
        max_scale = 1.3
        scale_step_size = 0.1
        crop_h, crop_w = crop_size, crop_size

        pad_value = tuple([int(v * 255) for v in mean_vals])
        ignore_label = (0, 0, 0)

        transform = T.Compose(
            [
                T.PhotometricDistort(),
                T.Resize(min_resize_value,
                         max_resize_value,
                         resize_factor),
                T.RandomScale(
                    min_scale,
                    max_scale,
                    scale_step_size
                ),
                T.RandomCrop(
                    crop_h,
                    crop_w,
                    pad_value,
                    ignore_label,
                    random_pad=True
                ),
                T.ToTensor(),
                T.RandomHorizontalFlip(),
                T.Normalize(
                    mean_vals,
                    std_vals
                )
            ]
        )
        ann_file = 'annotations/instances_train2017.json'
        data_root = '/home/ubuntu/hzx/data/MSCOCO2017'
        dataset = COCODataset(ann_file,
                              data_root,
                              pipeline=[],
                             transform=transform,
                             sup=args.sup,
                             sigma=args.sigma,
                             point_thresh=args.pseudo_thresh,
                              # min_gt_size=None,
                              # min_area_size=None,
                              args=args)
    else:
        ann_file = 'annotations/instances_val2017.json'
        data_root = 'data/coco/COCO2017'

        dataset = COCOTestDataset(ann_file,
                              data_root,
                              pipeline=[],
                             sup=args.sup,
                             sigma=args.sigma,
                             point_thresh=args.pseudo_thresh,
                              # min_gt_size=None,
                              # min_area_size=None
                                  )
    return dataset


class VOCDataset(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, 
                 transform=None, sup='cls', sigma=8, point_thresh=0.5, args=None):
        
        self.num_classes = num_classes
        self.transform = transform
        self.sigma = sigma
        self.sup = sup
        self.point_thresh = point_thresh
        
        self.g = gaussian(sigma)
        
        self.dat_list = self.read_labeled_image_list(root_dir, datalist_file)
        self.args = args

    def __getitem__(self, idx):
        img_path = self.dat_list["img"][idx]
        seg_map_path = self.dat_list["seg_map"][idx]
        cls_label = self.dat_list["cls_label"][idx]
        points = self.dat_list["point"][idx]
        
        img = np.uint8(Image.open(img_path).convert("RGB"))
        seg_map = np.uint8(Image.open(seg_map_path))
        
        if self.transform is not None:
            img, seg_map, points = self.transform(img, seg_map, points)

        center_map, offset_map, weight, masks = pseudo_label_generation(self.sup,
                                                                  seg_map.numpy(), 
                                                                  points, 
                                                                  cls_label, 
                                                                  self.num_classes, 
                                                                  self.sigma, 
                                                                  self.g, self.args)

        
        point_list = self.make_class_wise_point_list(points)
        
        seg_map = seg_map.long()
        center_map = torch.from_numpy(center_map)
        offset_map = torch.from_numpy(offset_map)
        weight = torch.from_numpy(weight)
        point_list = torch.from_numpy(point_list)
        masks = torch.from_numpy(masks)

        # aff = aff.cpu()
        # aff = torch.from_numpy(aff)

        return img, cls_label, seg_map, center_map, offset_map, weight, point_list, masks
        
        
    def make_class_wise_point_list(self, points):
        
        MAX_NUM_POINTS = 128
        
        point_list = np.zeros((self.num_classes, MAX_NUM_POINTS, 2), dtype=np.int32)
        point_count = [0 for _ in range(self.num_classes)]
        
        for (x, y, cls, _ ) in points:
            point_list[cls][point_count[cls]] = [y, x]
            point_count[cls] += 1
            
        return point_list
        
        
    def read_labeled_image_list(self, root_dir, data_list):
        img_dir = os.path.join(root_dir, "JPEGImages")
        seg_map_dir = os.path.join(root_dir, "SegmentationClass")
#         print('upperbound experiment.................')
        
        if self.sup == 'point':
            point_dir = os.path.join(root_dir, "Center_points")
        else:    
            point_dir = os.path.join(root_dir, "Peak_points")
            
        with open(data_list, 'r') as f:
            lines = f.read().splitlines()
            
        img_list = []
        label_list = []
        seg_map_list = []
        point_list = []
        
        np.random.shuffle(lines)
        
        for line in lines:
            fields = line.strip().split(" ")
            # fields[0] : file_name
            # fields[1:] : cls labels
            
            image_path = os.path.join(img_dir, fields[0] + '.jpg')
            seg_map_path = os.path.join(seg_map_dir, fields[0] + '.png')
            point_txt = os.path.join(point_dir, fields[0] + '.txt')
            
            # one-hot cls label
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
                
            # get points
            with open(point_txt, 'r') as pf:
                points = pf.read().splitlines()
                points = [p.strip().split(" ") for p in points]
                points = [ [float(p[0]), float(p[1]), int(p[2]), float(p[3])] for p in points if float(p[3]) > self.point_thresh]
                # point (x_coord, y_coord, class-idx, conf)
                
            img_list.append(image_path)
            seg_map_list.append(seg_map_path)
            point_list.append(points)
            label_list.append(labels)
            
        return {"img": img_list, 
                "cls_label": label_list, 
                "seg_map": seg_map_list, 
                "point": point_list}
    
    def __len__(self):
        return len(self.dat_list["img"])
        

class VOCTestDataset(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, args=None):
        self.num_classes = num_classes
        self.dat_list = self.read_labeled_image_list(root_dir, datalist_file)
        self.args = args

    def __getitem__(self, idx):
        fname = self.dat_list["fname"][idx]
        img_path = self.dat_list["img"][idx]
        cls_label = self.dat_list["cls_label"][idx]
        points = self.dat_list["point"][idx]
        
        img = Image.open(img_path).convert("RGB")
        
        ori_w, ori_h = img.size

        new_h = (ori_h + 31) // 32 * 32
        new_w = (ori_w + 31) // 32 * 32

        img = img.resize((new_w, new_h), Image.BILINEAR)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return img, cls_label, points, fname, (ori_h, ori_w)
        
    def read_labeled_image_list(self, root_dir, data_list):
        img_dir = os.path.join(root_dir, "JPEGImages")
        mask_dir = os.path.join(root_dir, "SegmentationObjectAug")
        point_dir = os.path.join(root_dir, "Center_points")
            
        with open(data_list, 'r') as f:
            lines = f.read().splitlines()
            
        fname_list = []
        img_list = []
        mask_list = []
        label_list = []
        point_list = []
        
        for line in lines:
            fields = line.strip().split(" ")
            # fields[0] : file_name
            # fields[1:] : cls labels
            
            image_path = os.path.join(img_dir, fields[0] + '.jpg')
            mask_path = os.path.join(mask_dir, fields[0] + '.png')
            point_txt = os.path.join(point_dir, fields[0] + '.txt')
            
            # one-hot cls label
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
                
            # get points
            with open(point_txt, 'r') as pf:
                points = pf.read().splitlines()
                points = [p.strip().split(" ") for p in points]
                points = [ [float(p[0]), float(p[1]), int(p[2]), float(p[3])] for p in points]
                # point (x_coord, y_coord, class-idx, conf)
                
                points_cls = [[] for _ in range(self.num_classes)]
                for (x, y, cls, _ ) in points:
                    points_cls[cls].append((y, x))
                
            fname_list.append(fields[0])
            img_list.append(image_path)
            mask_list.append(mask_path)
            point_list.append(points_cls)
            label_list.append(labels)
            
        return {"img": img_list, 
                "mask": mask_list,
                "cls_label": label_list, 
                "point": point_list, 
                "fname": fname_list}
    
    def __len__(self):
        return len(self.dat_list["img"])

class VOCTestAllPointDataset(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, args=None):
        self.num_classes = num_classes
        self.dat_list = self.read_labeled_image_list(root_dir, datalist_file)
        self.args = args

    def __getitem__(self, idx):
        fname = self.dat_list["fname"][idx]
        img_path = self.dat_list["img"][idx]
        cls_label = self.dat_list["cls_label"][idx]
        points_cls = self.dat_list["point_cls"][idx]
        points = self.dat_list["point"][idx]
        
        img = Image.open(img_path).convert("RGB")
        
        ori_w, ori_h = img.size

        new_h = (ori_h + 31) // 32 * 32
        new_w = (ori_w + 31) // 32 * 32

        img = img.resize((new_w, new_h), Image.BILINEAR)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return img, cls_label, points, points_cls, fname, (ori_h, ori_w)
        
    def read_labeled_image_list(self, root_dir, data_list):
        img_dir = os.path.join(root_dir, "JPEGImages")
        mask_dir = os.path.join(root_dir, "SegmentationObjectAug")
        point_dir = os.path.join(root_dir, "Center_points")
            
        with open(data_list, 'r') as f:
            lines = f.read().splitlines()
            
        fname_list = []
        img_list = []
        mask_list = []
        label_list = []
        point_list = []
        pointcls_list = []
        
        for line in lines:
            fields = line.strip().split(" ")
            # fields[0] : file_name
            # fields[1:] : cls labels
            
            image_path = os.path.join(img_dir, fields[0] + '.jpg')
            mask_path = os.path.join(mask_dir, fields[0] + '.png')
            point_txt = os.path.join(point_dir, fields[0] + '.txt')
            
            # one-hot cls label
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
                
            # get points
            with open(point_txt, 'r') as pf:
                points = pf.read().splitlines()
                points = [p.strip().split(" ") for p in points]
                points = [ [float(p[0]), float(p[1]), int(p[2]), float(p[3])] for p in points]
                # point (x_coord, y_coord, class-idx, conf)
                
                points_cls = [[] for _ in range(self.num_classes)]
                for (x, y, cls, _ ) in points:
                    points_cls[cls].append((y, x))
                
            fname_list.append(fields[0])
            img_list.append(image_path)
            mask_list.append(mask_path)
            pointcls_list.append(points_cls)
            point_list.append(points)
            label_list.append(labels)
            
        return {"img": img_list, 
                "mask": mask_list,
                "cls_label": label_list, 
                "point_cls": pointcls_list,
                "point": point_list,
                "fname": fname_list}
    
    def __len__(self):
        return len(self.dat_list["img"])

class VOCTestSelfDataset(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, sigma=8, args=None):

        self.num_classes = num_classes
        self.dat_list = self.read_labeled_image_list(root_dir, datalist_file)

        self.sigma = sigma
        self.sup = 'point'
        self.g = gaussian(sigma)
        self.args = args

    def __getitem__(self, idx):
        fname = self.dat_list["fname"][idx]
        img_path = self.dat_list["img"][idx]
        cls_label = self.dat_list["cls_label"][idx]
        points = self.dat_list["point"][idx]
        img = Image.open(img_path).convert("RGB")

        ori_w, ori_h = img.size

        new_h = (ori_h + 31) // 32 * 32
        new_w = (ori_w + 31) // 32 * 32

        img = img.resize((new_w, new_h), Image.BILINEAR)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        seg_map_path = self.dat_list["seg_map"][idx]
        seg_map = np.uint8(Image.open(seg_map_path))
        center_map, offset_map, weight = pseudo_label_generation(self.sup,
                                                                  seg_map,
                                                                  points,
                                                                  cls_label,
                                                                  self.num_classes,
                                                                  self.sigma,
                                                                  self.g, self.args)
        seg_map = seg_map.long()
        center_map = torch.from_numpy(center_map)
        offset_map = torch.from_numpy(offset_map)
        weight = torch.from_numpy(weight)

        return img, cls_label, seg_map, center_map, offset_map, points, fname, (ori_h, ori_w)

    def read_labeled_image_list(self, root_dir, data_list):
        img_dir = os.path.join(root_dir, "JPEGImages")
        mask_dir = os.path.join(root_dir, "SegmentationObjectAug")
        point_dir = os.path.join(root_dir, "Center_points")

        seg_map_dir = os.path.join(root_dir, "SegmentationClass")

        with open(data_list, 'r') as f:
            lines = f.read().splitlines()

        fname_list = []
        img_list = []
        mask_list = []
        label_list = []
        point_list = []

        seg_map_list = []

        for line in lines:
            fields = line.strip().split(" ")
            # fields[0] : file_name
            # fields[1:] : cls labels

            image_path = os.path.join(img_dir, fields[0] + '.jpg')
            mask_path = os.path.join(mask_dir, fields[0] + '.png')
            point_txt = os.path.join(point_dir, fields[0] + '.txt')

            seg_map_path = os.path.join(seg_map_dir, fields[0] + '.png')

            # one-hot cls label
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields) - 1):
                index = int(fields[i + 1])
                labels[index] = 1.

            # get points
            with open(point_txt, 'r') as pf:
                points = pf.read().splitlines()
                points = [p.strip().split(" ") for p in points]
                points = [[float(p[0]), float(p[1]), int(p[2]), float(p[3])] for p in points]
                # point (x_coord, y_coord, class-idx, conf)

                # points_cls = [[] for _ in range(self.num_classes)]
                # for (x, y, cls, _) in points:
                #     points_cls[cls].append((y, x))

            fname_list.append(fields[0])
            img_list.append(image_path)
            mask_list.append(mask_path)
            point_list.append(points)
            label_list.append(labels)

            seg_map_list.append(seg_map_path)

        return {"img": img_list,
                "mask": mask_list,
                "cls_label": label_list,
                "point": point_list,
                "fname": fname_list,
                "seg_map": seg_map_list
                }

    def __len__(self):
        return len(self.dat_list["img"])


from mmdet.datasets.cocofmt import CocoFmtDataset

class COCODataset(CocoFmtDataset):
    def __init__(self,
                 ann_file,
                 data_root=None,
                 transform=None,
                 sup='point',
                 sigma=6,
                 point_thresh=0.3,
                 # min_gt_size=None,
                 # min_area_size=None,
                 args=None,
                 **kwargs):
        self.transform = transform
        self.sigma = sigma
        self.sup = sup
        self.point_thresh = point_thresh
        self.g = gaussian(sigma)
        self.num_classes = 80
        self.args = args

        super(COCODataset, self).__init__(
            ann_file,
            data_root=data_root,
            # min_gt_size=min_gt_size,
            # min_area_size=min_area_size,
            **kwargs
        )
    def __getitem__(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)

        img_path = self.img_prefix + 'images/' + img_info['file_name']
        seg_map_path = self.img_prefix + 'coco_seg_anno/' + ann_info['seg_map']
        cls_label_temp = ann_info['labels']
        cls_label = np.zeros(len(self.CLASSES))
        cls_label[cls_label_temp] = 1

        points = [[float((ann_info['bboxes'][i][0] + ann_info['bboxes'][i][2]) / 2), float((ann_info['bboxes'][i][1] + ann_info['bboxes'][i][3]) / 2),  \
                   int(ann_info['labels'][i]), 1.0] for i in range(len(ann_info['bboxes']))]


        # for i in range(len(points)):
        #     points[i].append(cls_label_temp[i])
        #     points[i].append(1.0)
        img = np.uint8(Image.open(img_path).convert("RGB"))
        seg_map = np.uint8(Image.open(seg_map_path))

        if self.transform is not None:
            img, seg_map, points = self.transform(img, seg_map, points)

        center_map, offset_map, weight, masks = pseudo_label_generation(self.sup,
                                                                 seg_map.numpy(),
                                                                 points,
                                                                 cls_label,
                                                                 len(self.CLASSES),
                                                                 self.sigma,
                                                                 self.g,
                                                                 self.args)

        point_list = self.make_class_wise_point_list(points)

        seg_map = seg_map.long()
        center_map = torch.from_numpy(center_map)
        offset_map = torch.from_numpy(offset_map)
        weight = torch.from_numpy(weight)
        point_list = torch.from_numpy(point_list)
        masks = torch.from_numpy(masks)

        return img, cls_label, seg_map, center_map, offset_map, weight, point_list, masks

    def make_class_wise_point_list(self, points):

        MAX_NUM_POINTS = 128

        point_list = np.zeros((self.num_classes, MAX_NUM_POINTS, 2), dtype=np.int32)
        point_count = [0 for _ in range(self.num_classes)]

        for (x, y, cls, _) in points:
            point_list[cls][point_count[cls]] = [y, x]
            point_count[cls] += 1

        return point_list

    # def load_annotations(self, ann_file):
    #     """Load annotation from COCO style annotation file.
    #
    #     Args:
    #         ann_file (str): Path of annotation file.
    #
    #     Returns:
    #         list[dict]: Annotation info from COCO api.
    #     """
    #
    #     self.coco = COCO(ann_file)
    #     # The order of returned `cat_ids` will not
    #     # change with the order of the CLASSES
    #     self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
    #
    #     self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
    #     self.img_ids = self.coco.get_img_ids()
    #     data_infos = []
    #     total_ann_ids = []
    #     for i in self.img_ids:
    #         info = self.coco.load_imgs([i])[0]
    #         info['filename'] = info['file_name']
    #         data_infos.append(info)
    #         ann_ids = self.coco.get_ann_ids(img_ids=[i])
    #         total_ann_ids.extend(ann_ids)
    #     assert len(set(total_ann_ids)) == len(
    #         total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
    #     return data_infos


class COCOTestDataset(CocoFmtDataset):
    def __init__(self,
                 ann_file,
                 data_root=None,
                 transform=None,
                 sup='point',
                 sigma=6,
                 point_thresh=0.3,
                 # min_gt_size=None,
                 # min_area_size=None,
                 **kwargs):
        self.transform = transform
        self.sigma = sigma
        self.sup = sup
        self.point_thresh = point_thresh
        self.g = gaussian(sigma)
        self.num_classes = 80

        super(COCOTestDataset, self).__init__(
            ann_file,
            data_root=data_root,
            # min_gt_size=min_gt_size,
            # min_area_size=min_area_size,
            **kwargs
        )

    def __getitem__(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        fname = img_info['file_name'].split('.')[0]
        img_path = self.img_prefix + 'images/' + img_info['file_name']
        cls_label_temp = ann_info['labels']
        cls_label = np.zeros(len(self.CLASSES))
        cls_label[cls_label_temp] = 1
        points = [[float((ann_info['bboxes'][i][0] + ann_info['bboxes'][i][2]) / 2),
                   float((ann_info['bboxes'][i][1] + ann_info['bboxes'][i][3]) / 2), \
                   int(ann_info['labels'][i]), 1.0] for i in range(len(ann_info['bboxes']))]

        points_cls = [[] for _ in range(self.num_classes)]
        for (x, y, cls, _) in points:
            points_cls[cls].append((y, x))

        img = Image.open(img_path).convert("RGB")

        ori_w, ori_h = img.size

        new_h = (ori_h + 31) // 32 * 32
        new_w = (ori_w + 31) // 32 * 32

        img = img.resize((new_w, new_h), Image.BILINEAR)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return img, cls_label, points_cls, fname, (ori_h, ori_w)


if __name__ == '__main__':
    data_list = "data/val_cls.txt"
    dataset = VOCTestSelfDataset(data_list,
                                 root_dir='data/VOC2012',
                                 num_classes=20)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=n_gpus, rank=args.local_rank)
    valid_loader = DataLoader(dataset,
                              batch_size=1,
                              num_workers=0,
                              pin_memory=True,
                              sampler=valid_sampler,
                              drop_last=False)
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