# coding: utf-8

import torch
import numpy as np
from torch import nn
from torch.nn import functional
from torchvision import models

from .DenseNet import DenseNet121
from .BasicModule import BasicModule


class DenseNet121_RNN(BasicModule):
    def __init__(self, num_classes):
        super(DenseNet121_RNN, self).__init__()
        model = DenseNet121(num_classes=num_classes)

