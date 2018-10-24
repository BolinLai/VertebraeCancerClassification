# coding: utf-8

import os
import torch
import numpy as np
from torch import nn
from torch.nn import functional
from torchvision import models

from .pretrained import vgg19
from .BasicModule import BasicModule


class Vgg19(BasicModule):
    def __init__(self, num_classes):
        super(Vgg19, self).__init__()

        self.features = vgg19.features

        self.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=4096, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(in_features=4096, out_features=4096, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(in_features=4096, out_features=num_classes, bias=True))

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out

