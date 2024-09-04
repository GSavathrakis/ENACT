# ENACT: Entropy-based Attention Clustering for detection Transformers
This is the official implementation of the paper ENACT: Entropy-based Clustering of Attention Input for Improving the Computational Performance of Object Detection Transformers\
It is a plug-in module, used for clustering the input of Detection Transformers, based on their entropy which is learnable. In its current state, it can be plugged only in Detection Transformers that have a Multi-Head Self-Attention module in their encoder.\
In this repository, we plug ENACT to three such models, which are the [DETR](https://github.com/facebookresearch/detr), [Conditional DETR](https://github.com/Atten4Vis/ConditionalDETR) and [Anchor DETR](https://github.com/megvii-research/AnchorDETR).

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

## Instructions
Initially, clone the repository.
```
git clone https://github.com/GSavathrakis/ENACT.git
cd ENACT
```