# ENACT: Entropy-based Attention Clustering for detection Transformers
This is the official implementation of the paper ENACT: Entropy-based Clustering of Attention Input for Improving the Computational Performance of Object Detection Transformers\
It is a plug-in module, used for clustering the input of Detection Transformers, based on their entropy which is learnable. In its current state, it can be plugged only in Detection Transformers that have a Multi-Head Self-Attention module in their encoder.\
In this repository, we plug ENACT to three such models, which are the [DETR](https://github.com/facebookresearch/detr), [Conditional DETR](https://github.com/Atten4Vis/ConditionalDETR) and [Anchor DETR](https://github.com/megvii-research/AnchorDETR).\
\
We provide comparisons in GPU memory usage, training and inference times (in seconds per image) between detection transformer models, with and without ENACT. 
<table>
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>backbone</th>
      <th>epochs</th>
      <th>batch</th>
      <th>GPU (GB)</th>
      <th>train time</th>
      <th>inf time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DETR-C5</td>
      <td>R50</td>
      <td>300</td>
      <td>8</td>
      <td>36.5</td>
      <td>0.0541</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DETR-C5 + ENACT</td>
      <td>R50</td>
      <td>300</td>
      <td>8</td>
      <td>23.5</td>
      <td>0.0488</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Conditional DETR-C5</td>
      <td>R101</td>
      <td>50</td>
      <td>8</td>
      <td>46.6</td>
      <td>0.0826</td>
      <td>0.0637</td>
    </tr>
    <tr>
      <td>Conditional DETR-C5 + ENACT</td>
      <td>R101</td>
      <td>50</td>
      <td>8</td>
      <td>36.7</td>
      <td>0.0779</td>
      <td>0.0605</td>
    </tr>
    <tr>
      <td>Anchor DETR-DC5</td>
      <td>R50</td>
      <td>50</td>
      <td>4</td>
      <td>29.7</td>
      <td>0.0999</td>
      <td>0.0712</td>
    </tr>
    <tr>
      <td>Anchor DETR-DC5 + ENACT</td>
      <td>R50</td>
      <td>50</td>
      <td>4</td>
      <td>17.7</td>
      <td>0.0845</td>
      <td>0.0616</td>
    </tr>
  </tbody>
</table>
All experiments were done using the COCO 2017 train118k set for training, and val5k for validation. The precisions are computed based on the validation performance. We also provide logs and checkpoints for the models trained using ENACT.
<table>
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>AP</th>
      <th>APS</th>
      <th>APM</th>
      <th>APL</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DETR-C5</td>
      <td>40.6</td>
      <td>19.9</td>
      <td>44.3</td>
      <td>60.2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DETR-C5 + ENACT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Conditional DETR-C5</td>
      <td>42.8</td>
      <td>21.7</td>
      <td>46.6</td>
      <td>60.9</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Conditional DETR-C5 + ENACT</td>
      <td>41.5</td>
      <td>21.3</td>
      <td>45.5</td>
      <td>59.3</td>
      <td><a href="https://drive.google.com/file/d/1_RyhT_xn9TqqJy1-4mb39KUexjJrMV_d/view?usp=drive_link">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1OLhlMNd2k7g9lIb7mbWg82gkdeykAG6E/view?usp=drive_link">log</a></td>
    </tr>
    <tr>
      <td>Anchor DETR-DC5</td>
      <td>44.3</td>
      <td>25.1</td>
      <td>48.1</td>
      <td>61.1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Anchor DETR-DC5 + ENACT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>



## Instructions
Initially, clone the repository.
```
git clone https://github.com/GSavathrakis/ENACT.git
cd ENACT
```

## Acknowledgements

<details><summary> Expand </summary>
 
  * [https://github.com/facebookresearch/detr](https://github.com/facebookresearch/detr)
  * [https://github.com/megvii-research/AnchorDETR](https://github.com/megvii-research/AnchorDETR)
  * [https://github.com/Atten4Vis/ConditionalDETR](https://github.com/Atten4Vis/ConditionalDETR)
</details>