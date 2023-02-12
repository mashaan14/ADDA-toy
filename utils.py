"""Utilities for ADDA."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    # for every Conv layer in a model.
    if layer_name.find("Conv") != -1:
        # apply a normal distribution to the weights
        layer.weight.data.normal_(0.0, 0.02)

    # for every BatchNorm layer in a model.
    elif layer_name.find("BatchNorm") != -1:
        # apply a normal distribution to the weights
        layer.weight.data.normal_(1.0, 0.02)
        # apply bias = 0
        layer.bias.data.fill_(0)
