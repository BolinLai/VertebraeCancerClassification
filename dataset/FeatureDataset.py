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
    def __init__(self, root, csv_path, phase, trans=None, balance=True):
        """

        :param root:
        :param trans:
        :param phase:
        """
        with open(csv_path, 'r') as f:
            d = f.readlines()[1:]  # [1:]的作用是去掉表头
            features, labels = [], []
            print("Preparing {} data:".format(phase))
            for x in tqdm(d):
                image_path = os.path.join(root, str(x).strip().split(',')[0])
                label = int(str(x).strip().split('/')[-1][8])
                if phase == 'train' or phase == 'val':
                    # if label in [0, 1, 3]:
                    #     features.append(image_path)
                    #     labels.append(label)
                    features.append(image_path)
                    labels.append(label)
                elif phase == 'test':
                    # if label in [0, 1, 3]:
                    #     features.append(image_path)
                    #     labels.append(label)
                    features.append(image_path)
                    labels.append(label)
                elif phase == 'test_output':  # 只用于将每一张test data里的图片的预测结果和label输入到一个文件中
                    features.append(image_path)
                    labels.append(label)
                else:
                    raise ValueError
        f.close()

        self.features = features
        self.labels = labels
        self.phase = phase

        # a = []
        # for i in features:
        #     if i.split('/')[6] not in a:
        #         a.append(i.split('/')[6])
        # print('a:', len(a))

        feature_lists, label_lists = [], []
        f_tmp, l_tmp = [], []
        for f, l in zip(self.features, self.labels):
            if f_tmp and l_tmp:
                if f.split('/')[6] == f_tmp[-1].split('/')[6] and l != 2:
                    f_tmp.append(f)
                    l_tmp.append(l)
                elif f.split('/')[6] != f_tmp[-1].split('/')[6] and l != 2:
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

        feature_lists.append(f_tmp)
        label_lists.append(l_tmp)

        # 如果不剔除label=2
        # for f, l in zip(self.features, self.labels):
        #     if f_tmp and l_tmp:
        #         if f.split('/')[6] == f_tmp[-1].split('/')[6]:
        #             f_tmp.append(f)
        #             l_tmp.append(l)
        #         elif f.split('/')[6] != f_tmp[-1].split('/')[6]:
        #             feature_lists.append(f_tmp)
        #             label_lists.append(l_tmp)
        #             f_tmp, l_tmp = [f], [l]
        #         else:
        #             pass
        #     else:
        #         f_tmp.append(f)
        #         l_tmp.append(l)
        #
        # feature_lists.append(f_tmp)
        # label_lists.append(l_tmp)

        self.feature_lists = feature_lists
        self.label_lists = label_lists

        # print(len(self.feature_lists))
        # print(len(self.label_lists))

        # if trans is None:
        #     if phase == 'train':
        #         self.trans = transforms.Compose([
        #             transforms.RandomHorizontalFlip(),
        #             transforms.RandomVerticalFlip(),
        #             transforms.RandomRotation(30),
        #             transforms.ToTensor(),
        #             transforms.Lambda(lambda x: torch.cat([x]*3, 0)),
        #             transforms.Normalize(mean=VERTEBRAE_MEAN, std=VERTEBRAE_STD),
        #             # transforms.Lambda(lambda x: x * torch.Tensor(IMAGENET_STD).unsqueeze(1).unsqueeze(2) + torch.Tensor(IMAGENET_MEAN).unsqueeze(1).unsqueeze(2))
        #         ])
        #     elif phase == 'val' or phase == 'test' or phase == 'test_output':
        #         self.trans = transforms.Compose([
        #             transforms.ToTensor(),
        #             transforms.Lambda(lambda x: torch.cat([x]*3, 0)),
        #             transforms.Normalize(mean=VERTEBRAE_MEAN, std=VERTEBRAE_STD),
        #             # transforms.Lambda(lambda x: x * torch.Tensor(IMAGENET_STD).unsqueeze(1).unsqueeze(2) + torch.Tensor(IMAGENET_MEAN).unsqueeze(1).unsqueeze(2))
        #         ])
        #     else:
        #         raise IndexError

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
            elif i == 3:
                labels.append('R')
            else:
                raise ValueError

        # print(features.size())

        return features, labels

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
    train_data = Feature_Dataset('/DATA5_DB8/data/bllai/Data', '/DB/rhome/bllai/PyTorchProjects/Vertebrae/feature_train_path.csv', phase='train')
    val_data = Feature_Dataset('/DATA5_DB8/data/bllai/Data', '/DB/rhome/bllai/PyTorchProjects/Vertebrae/feature_test_path.csv', phase='val')

    # test_data = Vertebrae_Dataset('/DATA/data/hyguan/liuyuan_spine/data_all/patient_image_4', '/DB/rhome/bllai/PyTorchProjects/Vertebrae/test_path.csv', phase='test')
    # print(train_data.dist())
    # print(test_data.dist())
    l = []
    # train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)  # batch只能为1
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=4)  # batch只能为1

    for i, (f, l) in tqdm(enumerate(val_dataloader)):
        print(f.size())
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
