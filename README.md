# ADDA PyTorch implementation with a toy example

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.13-brightgreen.svg)](https://pytorch.org/get-started/previous-versions/)
[![torchvision](https://img.shields.io/badge/torchvision-0.14-brightgreen.svg)](https://pypi.org/project/torchvision/)

**A**dversarial **D**iscriminative **D**omain **A**daptation **(ADDA)** is one of the well-known benchmarks for domain adaptation tasks. **ADDA** was introduced in this paper:

```bibtex
@InProceedings{Tzeng_2017_CVPR,
  author =    {Tzeng, Eric and Hoffman, Judy and Saenko, Kate and Darrell, Trevor},
  title =     {Adversarial Discriminative Domain Adaptation},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year =      {2017},
  month =     {July}
}
```

This **ADDA** implementation uses a 2D toy dataset with built-in plots that help to visualize how the **ADDA** algorithm adapts the target features.

## Two dimensional dataset
The code starts by retrieving `source dataset` from data folder. Then it performs a rotation (domain shift) on a copy of the dataset. The rotated dataset is the `target dataset`. Here is a visualization of source and target datasets:
<p align="center">
  <img width="1200" src=dataset.png>
</p>

## Source domain classifier
The `encoder` and `classifier` networks are trained to separate `source class 0` and `source class 1`. Most of this logic happens in `core.train_src` function. Then, the learned model is tested on the test data:

<p align="center">
  <img width="1200" src="Testing source data using source encoder.png">
</p>

## Adversarial adaptation
The adversarial adaptation takes place in `core.train_tgt` function. The goal is to confuse the `discriminator` so it cannot tell if the sample is drawn from source or target domain. Once we reach this level of learning, we use this learned features to train the target encoder. For comparison, these features are passed through source classifier and target classifier:

```
>>> Testing target data using source encoder <<<
Avg Loss = 0.41065776348114014, Avg Accuracy = 89.000000%, ARI = 0.60646
```
<p align="center">
  <img width="1200" src="Testing target data using source encoder.png">
</p>

```
>>> Testing target data using target encoder <<<
Avg Loss = 0.3132730381829398, Avg Accuracy = 100.000000%, ARI = 1.00000
```
<p align="center">
  <img width="1200" src="Testing target data using target encoder.png">
</p>
