# coding: utf-8

import torch
import numpy as np
from torch import nn
from torch.nn import functional

from .pretrained import densenet121, chex_densenet121
from .BasicModule import BasicModule


class MultiTask_DenseNet121(BasicModule):
    def __init__(self, num_classes):
        super(MultiTask_DenseNet121, self).__init__()

        self.features = nn.Sequential(*list(densenet121.features.children()))

        self.adap_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier_1 = nn.Linear(1024, num_classes)

        self.classifier_2 = nn.Linear(1024, num_classes)

        # self.dropout = nn.Dropout(p=0.9)

    def forward(self, x):
        features = self.features(x)
        out = functional.relu(features, inplace=True)
        out = self.adap_avg_pool(out).view(features.size(0), -1)
        # out = functional.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        # out = self.dropout(out)

        out_1 = self.classifier_1(out)
        out_2 = self.classifier_2(out)
        return out_1, out_2

    def get_config_optim(self, lr_pre):
        return [{'params': self.features.parameters(), 'lr': lr_pre},
                {'params': self.adap_avg_pool.parameters()},
                {'params': self.classifier.parameters()}]


class CheXPre_MultiTask_DenseNet121(BasicModule):
    def __init__(self, num_classes):
        super(CheXPre_MultiTask_DenseNet121, self).__init__()

        self.features = chex_densenet121.densenet121.features

        self.adap_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # self.classifier_1 = nn.Linear(1024, num_classes)
        # self.classifier_2 = nn.Linear(1024, num_classes)
        self.classifier_1 = nn.Sequential(nn.Linear(1024, num_classes), nn.Sigmoid())
        self.classifier_2 = nn.Sequential(nn.Linear(1024, num_classes), nn.Sigmoid())

        # self.dropout = nn.Dropout(p=0.9)

    def forward(self, x):
        features = self.features(x)
        out = functional.relu(features, inplace=True)
        out = self.adap_avg_pool(out).view(features.size(0), -1)
        # out = functional.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        # out = self.dropout(out)

        out_1 = self.classifier_1(out)
        out_2 = self.classifier_2(out)
        return out_1, out_2
