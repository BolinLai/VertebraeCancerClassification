# coding: utf-8

import os
import random
import numpy as np
import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from pprint import pprint

VERTEBRAE_MEAN = [70.7] * 3
VERTEBRAE_STD = [181.5] * 3


class Lesion_Dataset(object):
    def __init__(self, root, csv_path, index_path, phase, trans=None):
        with open(csv_path, 'r') as f1, open(index_path, 'r') as f2:
            d1 = f1.readlines()[1:]
            d2 = f2.readlines()[1:]
        self.images = [os.path.join(root, x.strip().split(',')[0]) for x in d1]
        self.index = [(int(x.strip().split(',')[0]), int(x.strip().split(',')[1])) for x in d2]
        self.label = [int(x.strip().split(',')[2]) for x in d2]

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
            elif phase == 'val' or phase == 'test':
                self.trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: torch.cat([x]*3, 0)),
                    transforms.Normalize(mean=VERTEBRAE_MEAN, std=VERTEBRAE_STD)
                ])
            else:
                raise IndexError

    def __getitem__(self, index):
        image_index = self.index[index]
        image_path = self.images[image_index[0]: image_index[1]+1]
        image = [self.trans(Image.fromarray(np.load(p))) for p in image_path]
        # pprint(image_path)

        label = self.label[index]

        return image, label, image_path, image_index

    def __len__(self):
        return len(self.index)


if __name__ == '__main__':
    train_data = Lesion_Dataset(root='/DATA/data/hyguan/liuyuan_spine/data_all/patient_image_4',
                                csv_path='/DB/rhome/bllai/PyTorchProjects/Vertebrae/train_path.csv',
                                index_path='/DB/rhome/bllai/PyTorchProjects/Vertebrae/lesion_train_path.csv',
                                phase='train')
    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4)
    for i, (img, lab, img_path, img_index) in tqdm(enumerate(train_dataloader)):
        pprint(img[-1].size())
        print(lab)
        raise KeyboardInterrupt


