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


class MultiLabel_Dataset(object):
    def __init__(self, root, csv_path, phase, trans=None, balance=True):
        """

        :param root:
        :param trans:
        :param phase:
        """
        with open(csv_path, 'r') as f:
            d = f.readlines()[1:]  # [1:]的作用是去掉表头
            if phase == 'train' or phase == 'val':
                random.shuffle(d)
            else:
                pass
            if balance:
                images, labels = [], []
                normal_count = 0
                threshold = 3500 if phase == 'train' else 1230  # training data 取3500个无病的，validation data 取1230个，test data取所有的无病的
                print("Preparing balanced {} data:".format(phase))
                for x in tqdm(d):
                    image_path = os.path.join(root, str(x).strip().split(',')[0])
                    label = int(str(x).strip().split(',')[1])
                    if phase == 'train' or phase == 'val':
                        if label == 0 and normal_count < threshold:
                            images.append(image_path)
                            labels.append(label)
                            normal_count += 1
                        # elif label in [1, 2, 3]:
                        elif label in [1, 3]:
                            images.append(image_path)
                            labels.append(label)
                        else:
                            pass
                    elif phase == 'test':
                        if label in [0, 1, 3]:
                            images.append(image_path)
                            labels.append(label)
                    elif phase == 'test_output':  # 只用于将每一张test data里的图片的预测结果和label输入到一个文件中
                        images.append(image_path)
                        labels.append(label)
                    else:
                        raise ValueError
            else:
                images = [os.path.join(root, str(x).strip().split(',')[0]) for x in d]
                labels = [int(str(x).strip().split(',')[1]) for x in d]
        self.images = images
        self.labels = labels
        self.phase = phase

        if trans is None:
            if phase == 'train':
                self.trans = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(30),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: torch.cat([x]*3, 0)),
                    transforms.Normalize(mean=VERTEBRAE_MEAN, std=VERTEBRAE_STD)
                ])
            elif phase == 'val' or phase == 'test' or phase == 'test_output':
                self.trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: torch.cat([x]*3, 0)),
                    transforms.Normalize(mean=VERTEBRAE_MEAN, std=VERTEBRAE_STD)
                ])
            else:
                raise IndexError

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.fromarray(np.load(image_path))
        image = self.trans(image)

        label = self.labels[index]
        label_1 = 1 if label == 1 else 0
        label_2 = 1 if label == 3 else 0

        if label == 3 and self.phase != 'test_output':  # 做分类任务的label必须连续，不能使用[0,1,3]这种label，否则在混淆矩阵处会报错
            label = 2                                   # 对于'test_output'，需要将包括混合型在内的所有直接输出，所以不用调整

        return image, label_1, label_2, label, image_path

    def __len__(self):
        return len(self.images)

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
    train_data = MultiLabel_Dataset('/DATA/data/hyguan/liuyuan_spine/data_all/patient_image_4', '/DB/rhome/bllai/PyTorchProjects/Vertebrae/train_path.csv', phase='train', balance=True)
    # test_data = Vertebrae_Dataset('/DATA/data/hyguan/liuyuan_spine/data_all/patient_image_4', '/DB/rhome/bllai/PyTorchProjects/Vertebrae/test_path.csv', phase='test')
    # print(train_data.dist())
    # print(test_data.dist())

    l = []
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
    for i, (img, lab1, lab2, lab, img_path) in tqdm(enumerate(train_dataloader)):
        # raise KeyboardInterrupt
        l.append(img.squeeze())
    x = torch.cat(l, 0)
    print(x.size())
    print(x.mean(), x.std())


    # image = Image.fromarray(np.load('/DATA/data/hyguan/liuyuan_spine/data_all/patient_image_4/1/1695609/1695609_226/image.npy'))
    # trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[16], std=[127])])
    # image = trans(image)
    # print(image.mean(), image.std())
    # print(image.min(), image.max())
    # print(np.mean(image), np.std(image))
    # print(np.min(image), np.max(image))
