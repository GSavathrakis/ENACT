# ENACT
This is the official implementation of the paper ENACT: Entropy-based Clustering of Attention Input for Improving the Computational Performance of Object Detection Transformers\
It is a plug-in module, used for clustering the input of Detection Transformers, based on their entropy which is learnable. In its current state, it can be plugged only in Detection Transformers that have a Multi-Head Self-Attention module in their encoder.\
In this repository, we plug ENACT to three Detection Trnasformer models, which are the [DETR](https://github.com/facebookresearch/detr), [Conditional DETR](https://github.com/Atten4Vis/ConditionalDETR) and [Anchor DETR](https://github.com/megvii-research/AnchorDETR) 
