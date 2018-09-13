# coding: utf-8

import torch
from torch import nn
from torch.nn import functional
from torchvision import models

from .BasicModule import BasicModule


resnet34 = models.resnet34(pretrained=True)
resnet34.cuda()

densenet121 = models.densenet121(pretrained=True)
densenet121.cuda()


class CheXDenseNet121(BasicModule):
    def __init__(self, pretrained=False):
        super(CheXDenseNet121, self).__init__()

        self.densenet121 = densenet121

        kernelCount = densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, 14), nn.Sigmoid())

        if pretrained:
            self.load('/DATA5_DB8/data/sqpeng/Projects/chexnet/m-08082018-131619.pth.tar')

    def forward(self, x):
        features = self.densenet121.features(x)
        out = functional.relu(features, inplace=True)
        out = functional.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.densenet121.classifier(out)
        return out

    def load(self, path):
        state_dict = torch.load(path)['state_dict']
        keys = list(state_dict.keys())
        values = list(state_dict.values())

        for k, v in zip(keys, values):
            state_dict[k[7:]] = v
            del state_dict[k]

        self.load_state_dict(state_dict)


chex_densenet121 = CheXDenseNet121(pretrained=True)
chex_densenet121.cuda()
