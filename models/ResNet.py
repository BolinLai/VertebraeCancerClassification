# coding: utf-8

import torch
import numpy as np
from torch import nn
from torch.nn import functional
from torchvision import models

from .pretrained import resnet34
from .BasicModule import BasicModule


class ResNet34(BasicModule):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()

        self.features = nn.Sequential(resnet34.conv1,
                                      resnet34.bn1,
                                      resnet34.relu,
                                      resnet34.maxpool,
                                      resnet34.layer1,
                                      resnet34.layer2,
                                      resnet34.layer3,
                                      resnet34.layer4)

        self.adap_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = self.adap_avg_pool(features).view(features.size(0), -1)
        out = self.classifier(out)
        # print('out:', out.size())
        return out

    def get_config_optim(self, lr_pre):
        return [{'params': self.features.parameters(), 'lr': lr_pre},
                {'params': self.adap_avg_pool.parameters()},
                {'params': self.classifier.parameters()}]
