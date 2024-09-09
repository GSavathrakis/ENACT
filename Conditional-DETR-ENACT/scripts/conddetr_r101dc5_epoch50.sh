# ------------------------------------------------------------------------
# Copied from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env \
    main.py \
    --backbone resnet101 \
    --batch_size 1 \
    --dilation \
    --coco_path ../data/coco \
    --output_dir output/$script_name