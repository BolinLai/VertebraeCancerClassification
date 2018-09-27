# coding: utf-8

import os
import random
import numpy as np
import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VERTEBRAE_MEAN = [70.7] * 3
VERTEBRAE_STD = [181.5] * 3


class Feature_Dataset(object):
    def __init__(self, roots, csv_path, phase, trans=True, balance=True):
        """

        :param roots:
        :param trans:
        :param phase:
        """
        with open(csv_path, 'r') as f:
            d = f.readlines()[1:]  # [1:]的作用是去掉表头
        f.close()

        features, labels = [], []
        print("Preparing {} data:".format(phase))
        for root in roots:
            for x in tqdm(d):
                image_path = os.path.join(root, str(x).strip().split(',')[0])
                label = int(str(x).strip().split('/')[-1][8])
                features.append(image_path)
                labels.append(label)

        self.features = features
        self.labels = labels
        self.phase = phase

        feature_lists, label_lists = [], []
        f_tmp, l_tmp = [], []

        if self.phase != 'test_output':  # 剔除label=2
            for f, l in zip(self.features, self.labels):
                if f_tmp and l_tmp:
                    if f.split('/')[7] == f_tmp[-1].split('/')[7] and l != 2:
                        f_tmp.append(f)
                        l_tmp.append(l)
                    elif f.split('/')[7] != f_tmp[-1].split('/')[7] and l != 2:
                        feature_lists.append(f_tmp)
                        label_lists.append(l_tmp)
                        f_tmp, l_tmp = [f], [l]
                    else:
                        if f_tmp and l_tmp:
                            feature_lists.append(f_tmp)
                            label_lists.append(l_tmp)
                            f_tmp, l_tmp = [], []
                        else:
                            pass
                else:
                    if l != 2:
                        f_tmp.append(f)
                        l_tmp.append(l)
                    else:
                        pass
        else:  # 不剔除label=2
            for f, l in zip(self.features, self.labels):
                if f_tmp and l_tmp:
                    if f.split('/')[7] == f_tmp[-1].split('/')[7]:
                        f_tmp.append(f)
                        l_tmp.append(l)
                    elif f.split('/')[7] != f_tmp[-1].split('/')[7]:
                        feature_lists.append(f_tmp)
                        label_lists.append(l_tmp)
                        f_tmp, l_tmp = [f], [l]
                    else:
                        pass
                else:
                    f_tmp.append(f)
                    l_tmp.append(l)

        feature_lists.append(f_tmp)
        label_lists.append(l_tmp)

        self.feature_lists = feature_lists
        self.label_lists = label_lists

    def __getitem__(self, index):
        feature_list = self.feature_lists[index]
        label_list = self.label_lists[index]
        if len(feature_list) != len(label_list):
            raise ValueError

        features = [torch.unsqueeze(torch.from_numpy(np.load(p)), dim=0) for p in feature_list]
        features = torch.cat(features, dim=0)

        labels = []
        for i in label_list:
            if i == 0:
                labels.append('Z')
            elif i == 1:
                labels.append('C')
            elif i == 2:
                if self.phase == 'test_output':
                    labels.append('H')
                else:
                    raise ValueError
            elif i == 3:
                labels.append('R')
            else:
                raise ValueError

        # print(features.size())

        return features, labels, feature_list

    def __len__(self):
        return len(self.feature_lists)

    def dist(self):
        dist = {}
        print("Counting data distribution")
        for l in tqdm(self.labels):
            label = np.load(l)[0]
            if str(int(label)) in dist.keys():
                dist[str(int(label))] += 1
            else:
                dist[str(int(label))] = 1
        return dist


if __name__ == '__main__':
    # train_data = Feature_Dataset(roots=['/DATA5_DB8/data/bllai/Data/Features',
    #                                     '/DATA5_DB8/data/bllai/Data/Features_Horizontal_Vertical'],
    #                              csv_path='/DB/rhome/bllai/PyTorchProjects/Vertebrae/dataset/feature_train_path.csv',
    #                              phase='train')
    # train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)  # batch只能为1

    val_data = Feature_Dataset(roots=['/DATA5_DB8/data/bllai/Data/Features'],
                               csv_path='/DB/rhome/bllai/PyTorchProjects/Vertebrae/dataset/feature_test_path.csv',
                               phase='val')
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=4)  # batch只能为1

    for i, (f, l, fl) in tqdm(enumerate(val_dataloader)):
        pass
        # l.append(img.squeeze())
    # x = torch.cat(l, 0)
    # print(x.size())
    # print(x.mean(), x.std())


    # image = Image.fromarray(np.load('/DATA/data/hyguan/liuyuan_spine/data_all/patient_image_4/1/1695609/1695609_226/image.npy'))
    # trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[16], std=[127])])
    # image = trans(image)
    # print(image.mean(), image.std())
    # print(image.min(), image.max())
    # print(np.mean(image), np.std(image))
    # print(np.min(image), np.max(image))
