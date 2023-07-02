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
    
