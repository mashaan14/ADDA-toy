# ADDA PyTorch implementation with 2D toy example
Adversarial Discriminative Domain Adaptation (ADDA) is one of the well-known benchmarks for domain adaptation tasks. ADDA was introduced in this paper:

```
@InProceedings{Tzeng_2017_CVPR,
author = {Tzeng, Eric and Hoffman, Judy and Saenko, Kate and Darrell, Trevor},
title = {Adversarial Discriminative Domain Adaptation},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}
```

This ADDA implementation uses a 2D toy dataset with built-in plots that help to visualize how the ADDA algorithm is learning the new features.

## 2D dataset
The code starts by retrieving `source dataset` from data folder. Then it performs a rotation (domain shift) on a copy of the dataset. The rotated dataset is the `target dataset`. Here is a visualization of source and target datasets:
<p align="center">
  <img width="1200" src=dataset.png>
</p>

## Source domain classifier
The `encoder` and `classifier` networks are trained to separate `source class 0` and `source class 1`. Most of this logic happens in `core.train_src` function. Then, the learned model is tested on the test data:

```
Avg Loss = 0.12712214478670233, Avg Accuracy = 95.000000%
```

<p align="center">
  <img width="1200" src="Testing source data using source encoder.png">
</p>

## Adversarial adaptation
The adversarial adaptation takes place in `core.train_tgt` function. The goal is to confuse the ` discriminator` so it cannot tell if the sample is drawn from source or target domain. Once we reach this level of learning, we use this learned features to train the target encoder. For comparison, these features are passed through source classifier and target classifier:

```
>>> Testing target data using source encoder <<<
Avg Loss = 0.7519925472198606, Avg Accuracy = 83.000000%
```
<p align="center">
  <img width="1200" src="Testing target data using source encoder.png">
</p>

```
>>> Testing target data using target encoder <<<
Avg Loss = 0.20527904715205728, Avg Accuracy = 91.500000%
```
<p align="center">
  <img width="1200" src="Testing target data using target encoder.png">
</p>


## Code acknowledgement
I reused some code from this [repository](https://github.com/corenel/pytorch-adda).
