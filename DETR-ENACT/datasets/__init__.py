# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
#from .torchvision_datasets import CocoDetection
import torchvision

from .coco import build as build_coco
from .ShipRS import build as build_shiprs
from .hrsc import build as build_hrsc
from .dota import build as build_dota


def get_coco_api_from_dataset(dataset):
    #print(isinstance(dataset, CocoDetection))
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    #if isinstance(dataset, CocoDetection):
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'ShipRS':
        return build_shiprs(image_set, args)
    if args.dataset_file == 'hrsc':
        return build_hrsc(image_set, args)
    if args.dataset_file == 'dota':
        return build_dota(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
