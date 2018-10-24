# coding: utf-8

import os
import torch
import numpy as np
from torch import nn
from torch.nn import functional

from .pretrained import densenet121, vgg19
from .BasicModule import BasicModule


class MultiResDenseNet121(BasicModule):
    def __init__(self, num_classes):
        super(MultiResDenseNet121, self).__init__()

        self.prenet = nn.Sequential(densenet121.features.conv0,
                                    densenet121.features.norm0,
                                    densenet121.features.relu0,
                                    densenet121.features.pool0)

        self.dense1 = densenet121.features.denseblock1
        self.dense2 = densenet121.features.denseblock2
        self.dense3 = densenet121.features.denseblock3
        self.dense4 = densenet121.features.denseblock4

        self.trans1 = densenet121.features.transition1
        self.trans2 = densenet121.features.transition2
        self.trans3 = densenet121.features.transition3

        self.norm_shallow = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.norm_deep = densenet121.features.norm5

        self.adap_avg_pool_shallow = nn.AdaptiveAvgPool2d((1, 1))
        self.adap_avg_pool_deep = nn.AdaptiveAvgPool2d((1, 1))

        # self.classifier = nn.Linear(1024 + 256, num_classes)
        self.classifier = nn.Linear(1024, num_classes)

        # self.dropout = nn.Dropout(p=0.9)

    def forward(self, x):
        out = self.prenet(x)
        dense_out_1 = self.dense1(out)
        dense_out_2 = self.dense2(self.trans1(dense_out_1))
        dense_out_3 = self.dense3(self.trans2(dense_out_2))
        dense_out_4 = self.dense4(self.trans3(dense_out_3))
        out = self.norm_deep(dense_out_4)
        # print('dense_out_1 shape:', dense_out_1.size())
        # print('out shape:', out.size())

        # shallow_pool = self.adap_avg_pool_shallow(self.norm_shallow(dense_out_1)).view(dense_out_1.size(0), -1)
        deep_pool = self.adap_avg_pool_deep(functional.relu(out, inplace=True)).view(out.size(0), -1)
        # out = torch.cat([shallow_pool, deep_pool], dim=1)
        # print('shallow_pool', shallow_pool.size())
        # print('deep_pool', deep_pool.size())
        # print('out', out.size())

        # out = self.classifier(out)
        out = self.classifier(deep_pool)
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


class MultiResVgg19(BasicModule):
    def __init__(self, num_classes):
        super(MultiResVgg19, self).__init__()

        # vgg19_structure = list(vgg19.features.children())
        # self.conv1 = nn.Sequential(*list(vgg19_structure[:5]))
        # self.conv2 = nn.Sequential(*list(vgg19_structure[5:10]))
        # self.conv3 = nn.Sequential(*list(vgg19_structure[10:19]))
        # self.conv4 = nn.Sequential(*list(vgg19_structure[19:28]))
        # self.conv5 = nn.Sequential(*list(vgg19_structure[28:]))
        #
        # self.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=4096, bias=True),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Dropout(p=0.5),
        #                                 nn.Linear(in_features=4096, out_features=4096, bias=True),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Dropout(p=0.5),
        #                                 nn.Linear(in_features=4096, out_features=num_classes, bias=True))

        # vgg19_structure = list(vgg19.features.children())
        # self.conv1 = nn.Sequential(*list(vgg19_structure[:5]))
        # self.conv2 = nn.Sequential(*list(vgg19_structure[5:10]))
        # self.conv3 = nn.Sequential(*list(vgg19_structure[10:19]))
        # self.conv4 = nn.Sequential(*list(vgg19_structure[19:28]))
        # self.conv5 = nn.Sequential(*list(vgg19_structure[28:]))
        #
        # self.adap_avg_pool_5 = nn.AdaptiveAvgPool2d((1, 1))
        #
        # self.classifier = nn.Linear(512, num_classes)

        vgg19_structure = list(vgg19.features.children())
        self.conv1 = nn.Sequential(*list(vgg19_structure[:5]))
        self.conv2 = nn.Sequential(*list(vgg19_structure[5:10]))
        self.conv3 = nn.Sequential(*list(vgg19_structure[10:19]))
        self.conv4 = nn.Sequential(*list(vgg19_structure[19:28]))
        self.conv5 = nn.Sequential(*list(vgg19_structure[28:]))

        self.adap_avg_pool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.adap_avg_pool_5 = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(512 + 64, num_classes)

    def forward(self, x):
        # features = self.conv1(x)
        # features = self.conv2(features)
        # features = self.conv3(features)
        # features = self.conv4(features)
        # features = self.conv5(features)
        # features = features.view(features.size(0), -1)
        # out = self.classifier(features)

        # out_1 = self.conv1(x)
        # out_2 = self.conv2(out_1)
        # out_3 = self.conv3(out_2)
        # out_4 = self.conv4(out_3)
        # out_5 = self.conv5(out_4)
        #
        # out = self.adap_avg_pool_5(out_5).view(out_5.size(0), -1)
        # out = self.classifier(out)

        out_1 = self.conv1(x)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        out_5 = self.conv5(out_4)

        feature_1 = self.adap_avg_pool_1(out_1).view(out_1.size(0), -1)
        feature_5 = self.adap_avg_pool_5(out_5).view(out_5.size(0), -1)
        feature = torch.cat([feature_1, feature_5], dim=1)

        out = self.classifier(feature)

        return out
