import os
import fire
import pandas as pd
import numpy as np

from tqdm import tqdm
from pprint import pprint
from config import config
from utils import write_csv


def slice_wise(train_csv="sup_train_path.csv", test_csv="sup_test_path.csv"):
    files = os.listdir(config.data_root)
    train, test = [], []
    for file in sorted(files):
        if os.path.isdir(os.path.join(config.data_root, file)):
            if file in ['2', '3', '4', '5', '6', '7', '8', '9']:
                train.extend([os.path.join(file, x) for x in sorted(os.listdir(os.path.join(config.data_root, file)))])
            elif file == '1':
                test.extend([os.path.join(file, x) for x in sorted(os.listdir(os.path.join(config.data_root, file)))])
    print('Training patients:', len(train), 'Test patients:', len(test))
    print(train[0])

    train_image = [os.path.join(x, s, 'image.npy') for x in train for s in sorted(os.listdir(os.path.join(config.data_root, x)), key=lambda id: int(id.split('_')[1]))]
    train_label = [os.path.join(x, s, 'label.npy') for x in train for s in sorted(os.listdir(os.path.join(config.data_root, x)), key=lambda id: int(id.split('_')[1]))]
    test_image = [os.path.join(x, s, 'image.npy') for x in test for s in sorted(os.listdir(os.path.join(config.data_root, x)), key=lambda id: int(id.split('_')[1]))]
    test_label = [os.path.join(x, s, 'label.npy') for x in test for s in sorted(os.listdir(os.path.join(config.data_root, x)), key=lambda id: int(id.split('_')[1]))]
    print(len(test_image), len(test_label))
    print(test_image[34], test_label[34])

    train_label1, train_label2, train_label3, train_label4 = [], [], [], []
    test_label1, test_label2, test_label3, test_label4 = [], [], [], []
    for i in tqdm(train_label, desc='Making Training CSV'):
        label = np.load(os.path.join(config.data_root, i))
        train_label1.append(int(label[0]))
        train_label2.append(int(label[1]))
        train_label3.append(int(label[2]))
        train_label4.append(int(label[3]))

    for i in tqdm(test_label, desc='Making Test CSV'):
        label = np.load(os.path.join(config.data_root, i))
        test_label1.append(int(label[0]))
        test_label2.append(int(label[1]))
        test_label3.append(int(label[2]))
        test_label4.append(int(label[3]))

    dataframe = pd.DataFrame({'image': train_image, 'label1': train_label1, 'label2': train_label2, 'label3': train_label3, 'label4': train_label4})
    dataframe.to_csv(os.path.join('dataset', train_csv), index=False, sep=',')
    dataframe = pd.DataFrame({'image': test_image, 'label1': test_label1, 'label2': test_label2, 'label3': test_label3, 'label4': test_label4})
    dataframe.to_csv(os.path.join('dataset', test_csv), index=False, sep=',')


def feature_slice_wise(train_csv="feature_train_path.csv", test_csv="feature_test_path.csv"):
    files = os.listdir(os.path.join(config.data_root, 'Features'))
    train, test = [], []
    for file in files:
        if os.path.isdir(os.path.join(config.data_root, file)):
            if file in ['1', '2', '3']:
                train.extend([os.path.join(file, x) for x in sorted(os.listdir(os.path.join(config.data_root, file)))])
            elif file == '4':
                test.extend([os.path.join(file, x) for x in sorted(os.listdir(os.path.join(config.data_root, file)))])
    print(len(train), len(test))
    print(train[0])

    train_feature = [os.path.join(x, s, t)
                     for x in tqdm(train)
                     for s in
                     sorted(os.listdir(os.path.join(config.data_root, x)), key=lambda id: int(id.split('_')[1]))
                     for t in os.listdir(os.path.join(config.data_root, x, s))]
    test_feature = [os.path.join(x, s, t)
                    for x in tqdm(test)
                    for s in sorted(os.listdir(os.path.join(config.data_root, x)), key=lambda id: int(id.split('_')[1]))
                    for t in os.listdir(os.path.join(config.data_root, x, s))]

    print(len(train_feature), len(test_feature))
    print(test_feature[34])

    dataframe = pd.DataFrame({'feature': train_feature})
    dataframe.to_csv(os.path.join('dataset', train_csv), index=False, sep=',')
    dataframe = pd.DataFrame({'image': test_feature})
    dataframe.to_csv(os.path.join('dataset', test_csv), index=False, sep=',')


def lesion_wise(path):
    with open(path, 'r') as f:
        d = f.readlines()[1:]

    patient_id = d[0].strip().split(',')[0].split('/')[1]  # 第一张image的病人id
    lesion_label = int(d[0].strip().split(',')[1])  # 第一张image的label
    lesions = []  # 记录病灶的起始index
    start = 0

    for i, x in enumerate(d):
        patient = x.strip().split(',')[0].split('/')[1]
        label = int(x.strip().split(',')[1])

        if label != lesion_label or patient != patient_id:
            lesions.append((start, i - 1, lesion_label))
            start = i
            patient_id = patient
            lesion_label = label

    write_csv('lesion_' + path, tag=['start', 'end', 'label'], content=lesions)
    # pprint(sorted(lesions, key=lambda m: m[1]-m[0]))  # max silce number = 615


if __name__ == '__main__':
    fire.Fire({
        'slice_wise': slice_wise,
        'feature_slice_wise': feature_slice_wise,
    })
