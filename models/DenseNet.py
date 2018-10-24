# coding: utf-8

import os
import torch
import numpy as np
from torch import nn
from torch.nn import functional

from .pretrained import densenet121, chex_densenet121
from .BasicModule import BasicModule


class DenseNet121(BasicModule):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()

        self.features = nn.Sequential(*list(densenet121.features.children()))

        self.adap_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(1024, num_classes)

        # self.dropout = nn.Dropout(p=0.9)

    def forward(self, x):
        features = self.features(x)
        out = functional.relu(features, inplace=True)
        out = self.adap_avg_pool(out).view(features.size(0), -1)
        # out = functional.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        # out = self.dropout(out)

        out = self.classifier(out)
        return out

    def save_feature(self, x, target, image_path, feature_folder):
        features = self.features(x)
        out = functional.relu(features, inplace=True)
        out = self.adap_avg_pool(out).view(features.size(0), -1)
        out = out.data.cpu().numpy()
        for i in range(out.shape[0]):
            feature_folder_path = os.path.join(feature_folder, '/'.join(image_path[i].split('/')[7:-1]))
            if not os.path.exists(feature_folder_path):
                os.makedirs(feature_folder_path)
            np.save(os.path.join(feature_folder_path, 'feature_'+str(int(target[i]))+'.npy'), out[i])

    def get_config_optim(self, lr_pre):
        return [{'params': self.features.parameters(), 'lr': lr_pre},
                {'params': self.adap_avg_pool.parameters()},
                {'params': self.classifier.parameters()}]


class CheXPre_DenseNet121(BasicModule):
    def __init__(self, num_classes):
        super(CheXPre_DenseNet121, self).__init__()

        self.features = chex_densenet121.densenet121.features

        self.vertebrae_classifier = nn.Sequential(nn.Linear(1024, num_classes), nn.Sigmoid())
        # self.vertebrae_classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = functional.relu(features, inplace=True)
        out = functional.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.vertebrae_classifier(out)
        return out


# class CheXDenseNet121(BasicModule):
#     def __init__(self, num_classes):
#         super(CheXDenseNet121, self).__init__()
#
#         # self.features = model.features
#         self.densenet121 = model
#         # self.densenet121.cuda()
#
#         kernelCount = model.classifier.in_features
#         self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, 14), nn.Sigmoid())
#         # self.load(torch.load('/DATA5_DB8/data/sqpeng/Projects/chexnet/m-08082018-131619.pth.tar')['state_dict'])
#         # self.load('/DATA5_DB8/data/sqpeng/Projects/chexnet/m-08082018-131619.pth.tar')
#
#         self.vertebrae_classifier = nn.Sequential(nn.Linear(kernelCount, num_classes), nn.Sigmoid())
#         # self.classifier_1 = nn.Sequential(nn.Linear(kernelCount, num_classes), nn.Sigmoid())
#         # self.classifier_2 = nn.Sequential(nn.Linear(kernelCount, num_classes), nn.Sigmoid())
#
#     def forward(self, x):
#         # features = self.features(x)
#         features = self.densenet121.features(x)
#         out = functional.relu(features, inplace=True)
#         out = functional.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
#         out = self.vertebrae_classifier(out)
#         return out
#
#     # def load(self, path):
#     #     state_dict = torch.load(path)['state_dict']
#     #     keys = list(state_dict.keys())
#     #     values = list(state_dict.values())
#     #
#     #     for k, v in zip(keys, values):
#     #         state_dict[k[7:]] = v
#     #         del state_dict[k]
#     #
#     #     self.load_state_dict(state_dict)
