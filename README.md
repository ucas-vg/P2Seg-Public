# P2Seg - Official Pytorch Implementation (ICLR 2024)

**P2Seg: Pointly-supervised Segmentation via Mutual Distillation (ICLR 2024)** <br />
Zipeng Wang<sup>1</sup>, Xuehui Yu<sup>1</sup>, Xumeng Han<sup>1</sup>, Wenwen Yu<sup>1</sup>, Zhixun Huang<sup>2</sup>, Jianbin Jiao<sup>1</sup>, Zhenjun Han<sup>1</sup><br>

<sup>1</sup> <sub>University of Chinese Academy of Sciences</sub><br />
<sup>2</sup> <sub>Xiaomi AI Lab, Beijing, China</sub><br />

[![Paper](https://img.shields.io/badge/arXiv-2401.09709-brightgreen)](https://arxiv.org/abs/2401.09709)

<img src = "https://github.com/ucas-vg/P2Seg_Public/blob/main/P2Seg/figures/frame5.png" width="100%" height="100%">


# Abtract

Point-level Supervised Instance Segmentation (PSIS) aims to enhance the applicability and scalability of instance segmentation by utilizing low-cost yet instance-informative annotations. Existing PSIS methods usually rely on positional information to distinguish objects, but predicting precise boundaries remains challenging due to the lack of contour annotations. Nevertheless, weakly supervised semantic segmentation methods are proficient in utilizing intra-class feature consistency to capture the boundary contours of the same semantic regions.
In this paper, we design a **M**utual **D**istillation **M**odule (MDM) to leverage the complementary strengths of both instance position and semantic information and achieve accurate instance-level object perception. The MDM consists of **S**emantic to **I**nstance (S2I) and **I**nstance to **S**emantic (I2S). S2I is guided by the precise boundaries of semantic regions to learn the association between annotated points and instance contours. I2S leverages discriminative relationships between instances to facilitate the differentiation of various objects within the semantic map. 
Extensive experiments substantiate the efficacy of MDM in fostering the synergy between instance and semantic information, consequently improving the quality of instance-level object representations. Our method achieves 55.7 mAP<sub>50</sub> and 17.6 mAP on the PASCAL VOC and MS COCO datasets, significantly outperforming recent PSIS methods and several box-supervised instance segmentation competitors. The code is available at https://github.com/ucas-vg/P2Seg.

# Experimental Results (VOC 2012, COCO)

<img src = "https://github.com/ucas-vg/P2Seg/blob/main/P2Seg/figures/voc_val_set.png" width="50%" height="50%">
<img src = "https://github.com/ucas-vg/P2Seg/blob/main/P2Seg/figures/coco_val_set.png" width="50%" height="50%">
<img src = "https://github.com/ucas-vg/P2Seg/blob/main/P2Seg/figures/coco_test_dev.png" width="50%" height="50%">

* P2Seg (ResNet101, point-label) : 53.9 mAP50 on VOC2012 [[download]](https://drive.google.com/drive/folders/10_iwtNyaWmRRbG_GNF3V_J1E8T1tzf2l?usp=drive_link)
* P2Seg (HRNet48, point-label) : 55.6 mAP50 on VOC2012 [[download]](https://drive.google.com/drive/folders/10_iwtNyaWmRRbG_GNF3V_J1E8T1tzf2l?usp=drive_link)
* P2Seg (HRNet48, point-label) : 17.6 mAP50 on COCO2017 [[download]](https://drive.google.com/drive/folders/1nFOw2lITwZX6xC0MYGm8a1GPskGFiuW3?usp=drive_link)

# Qualitative Results
* COCO2017

<img src = "https://github.com/ucas-vg/P2Seg/blob/main/P2Seg/figures/vis_coco.jpeg" width="50%" height="50%">

* VOC2012

<img src = "https://github.com/ucas-vg/P2Seg/blob/main/P2Seg/figures/vis_voc.jpeg" width="50%" height="50%">

# News

- [x] official pytorch code release
- [x] update training code and dataset for VOC
- [x] update training code and dataset for COCO

# How To Run

### Requirements
- torch>=1.10.1
- torchvision>=0.11.2
- chainercv>=0.13.1
- numpy
- pillow
- scikit-learn
- tqdm

### Datasets

- Download Pascal VOC2012 dataset from the [official dataset homepage](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
- Download Pascal COCO2017 dataset from the [official dataset homepage](http://cocodataset.org).
- Download other data from [[here]](https://drive.google.com/drive/folders/1iMd4FQl23ZIkV9ZO6XptaSfL6BdKMrNz?usp=drive_link).
  - `train_SAM_point_top1.json` (COCO train ann_file)
  - `instances_train2017.json` (COCO validate ann_file)
  - `train_cls.txt` (VOC train data_list)
  - `val_cls.txt` (VOC validate data_list)

### Point Supervised Instance Segmentation on COCO2017(train)
```
# change the save_folder and result_dir; no need to change root_dir because we set it up in utils/LoadData.py
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--master_port=10025 --nproc_per_node=8 main.py --batch_size 8 --root_dir data/VOCdevkit/VOC2012 \
--dataset coco --num_classes 80 --sup 'point' --refine True --lr 5e-5 --train_iter 200000 --val_freq 200000 \
--val_thresh 0.1 --val_clean True --val_flip False --save_to_rle True --dis_group True --affinity_loss True \
--iterate_refine 1 --annotation_loss False --backbone hrnet48 --mode 'train' --save_folder wzp_result/ckpts/coco/sam_lr5e-5_semsam_addaff \
--result_dir wzp_result/ckpts/coco/sam_lr5e-5_semsam_addaff/result_pickle > wzp_result/ckpts/coco/sam_lr5e-5_semsam_addaff/BESTIE_log_train_addaff.txt
```

### Point Supervised Instance Segmentation on COCO2017(val)
```
# change the save_folder and result_dir; no need to change root_dir because we set it up in utils/LoadData.py
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--master_port=10006 --nproc_per_node=8 main.py --batch_size 8 --root_dir data/VOCdevkit/VOC2012   \
--dataset coco --num_classes 80 --sup 'point' --refine True --lr 5e-5 --train_iter 200000 --val_freq 200000  \
--val_thresh 0.1 --val_clean True --val_flip False --save_to_rle True   --dis_group True  --affinity_loss True  \
--iterate_refine 1 --annotation_loss False  --backbone hrnet48  --mode 'validate'  \
--save_folder ckpts/coco/sam_lr5e-5_semsam   --result_dir  ckpts/coco/sam_lr5e-5_semsam/result_pickle_test \
--resume ckpts/coco/sam_lr5e-5_semsam/last.pt
```

### Image-level Supervised Instance Segmentation on VOC2012(train)
```
# change the save_folder and result_dir
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--master_port=10143 --nproc_per_node=8 main.py --batch_size 8 --root_dir /home/ubuntu/hzx/ijcv/TOV_mmdetection/projects/BESTIE/data/VOCdevkit/VOC2012   \
--dataset voc --num_classes 20 --sup 'point' --refine True --lr 5e-5 --train_iter 50000 --val_freq 1000 --num_workers 1  \
--center_weight 0.0 --val_thresh 0.1 --val_clean True --val_flip False --save_to_rle True --dis_group True --affinity_loss True  \
--iterate_refine 1 --annotation_loss False --backbone hrnet48 --mode 'train' --random_seed 3407 \
--save_folder /home/ubuntu/hzx/ijcv/BESTIE_sam/wzp_result/ckpts/voc/base_ppg_aff_affrefine_gflops \
--result_dir /home/ubuntu/hzx/ijcv/BESTIE_sam/wzp_result/ckpts/voc/base_ppg_aff_affrefine_gflops/result_pickle_allpoint_32 > /home/ubuntu/hzx/ijcv/BESTIE_sam/wzp_result/ckpts/voc/base_ppg_aff_affrefine_gflops/base_ppg_aff_affrefine.txt
```

### Image-level Supervised Instance Segmentation on VOC2012(val)
```
# change the --mode ‘train’ to --mode ‘validate’
```

### Mask R-CNN Refinement

1. Generate COCO-style pseudo labels using the BESTIE model. You can download the version we have generated  from [[here]](https://drive.google.com/drive/folders/1oeNwq7j-pyUWuTrFeEAR4Gy047_LcEy5?usp=drive_link).
2. Train the Mask R-CNN using the pseudo-labels: https://github.com/ucas-vg/TOV_mmdetection .
```
# change the exp, work-dir and data.train.ann_file 
GPU=8 && LR=0.02 && B=2 && exp='configs3/COCO/mask_rcnn_x101_fpn_1x_coco_aug' && \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && PORT=10000   tools/dist_train.sh ${exp}.py ${GPU} \
--work-dir ../TOV_mmdetection_wzp_result/work_dir/${exp}/sam_samsem_200000_test_dev/ \
--cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B} \
data.train.ann_file='/home/ubuntu/hzx/ijcv/BESTIE_sam/wzp_result/ckpts/coco/sam_lr5e-5_semsam_addaff/voc_data_test/coco_ins_train_cls_refine_90.json'
```

## Citation

And if the following works do some help for your research, please cite:
```
@inproceedings{P2Seg,
  author    = {Zipeng Wang, Xuehui Yu, Xumeng Han, Wenwen Yu, Zhixun Huang, Jianbin Jiao, Zhenjun Han},
  title     = {P2Seg: Pointly-supervised Segmentation via Mutual Distillation},
  booktitle = {ICLR},
  year      = {2024},
}
```

# License

```
MIT License

Copyright (c) 2024 Vision Group, EECE of UCAS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

