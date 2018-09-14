# coding: utf-8
import os
import fire
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter

from config import config
from dataset import Feature_Dataset
from models import BiLSTM_CRF, START_TAG, STOP_TAG
from utils import Visualizer, write_csv, calculate_index

EMBEDDING_DIM = 1024
HIDDEN_DIM = 20
NUM_LAYERS = 5


def train(**kwargs):
    config.parse(kwargs)
    vis = Visualizer(port=2333, env=config.env)

    train_data = Feature_Dataset(config.data_root, config.train_paths, phase='train', balance=config.data_balance)
    val_data = Feature_Dataset(config.data_root, config.test_paths, phase='val', balance=config.data_balance)
    print('Training Feature Lists:', train_data.__len__(), 'Validation Feature Lists:', val_data.__len__())

    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=config.num_workers)

    tag_to_ix = {"Z": 0, "C": 1, "R": 2, START_TAG: 3, STOP_TAG: 4}

    # prepare model
    model = BiLSTM_CRF(tag_to_ix=tag_to_ix, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)

    if config.load_model_path:
        model.load(config.load_model_path)
    if config.use_gpu:
        model.cuda()

    model.train()

    # criterion and optimizer
    lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # metric
    loss_meter = meter.AverageValueMeter()
    previous_loss = 100000
    previous_acc = 0

    # train
    if not os.path.exists(os.path.join('checkpoints', model.model_name)):
        os.mkdir(os.path.join('checkpoints', model.model_name))

    for epoch in range(config.max_epoch):
        loss_meter.reset()
        train_cm = [[0]*3, [0]*3, [0]*3]
        count = 0

        # train
        for i, (features, labels, feature_paths) in tqdm(enumerate(train_dataloader)):
            # prepare input
            target = torch.LongTensor([tag_to_ix[t[0]] for t in labels])

            feat = Variable(features.squeeze())
            # target = Variable(target)
            if config.use_gpu:
                feat = feat.cuda()
                # target = target.cuda()

            model.zero_grad()

            try:
                neg_log_likelihood = model.neg_log_likelihood(feat, target)
            except NameError:
                count += 1
                continue

            neg_log_likelihood.backward()
            optimizer.step()

            loss_meter.add(neg_log_likelihood.data[0])
            result = model(feat)
            for t, r in zip(target, result[1]):
                train_cm[t][r] += 1

            if i % config.print_freq == config.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])
                print('loss', loss_meter.value()[0])

        train_accuracy = 100. * sum([train_cm[c][c] for c in range(config.num_classes)]) / np.sum(train_cm)
        val_cm, val_accuracy, val_loss = val(model, val_dataloader, tag_to_ix)

        if val_accuracy > previous_acc:
            if config.save_model_name:
                model.save(os.path.join('checkpoints', model.model_name, config.save_model_name))
            else:
                model.save(os.path.join('checkpoints', model.model_name, model.model_name+'_best_model.pth'))
            previous_acc = val_accuracy

        vis.plot_many({'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy})
        vis.log("epoch: [{epoch}/{total_epoch}], lr: {lr}, loss: {loss}".format(
            epoch=epoch + 1, total_epoch=config.max_epoch, lr=lr, loss=loss_meter.value()[0]))
        vis.log('train_cm:')
        vis.log(train_cm)
        vis.log('val_cm')
        vis.log(val_cm)
        print('train_accuracy:', train_accuracy, 'val_accuracy:', val_accuracy)
        print("epoch: [{epoch}/{total_epoch}], lr: {lr}, loss: {loss}".format(
            epoch=epoch + 1, total_epoch=config.max_epoch, lr=lr, loss=loss_meter.value()[0]))
        print('train_cm:')
        print(train_cm)
        print('val_cm:')
        print(val_cm)

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * config.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]


def val(model, dataloader, tag_to_ix):
    model.eval()
    val_cm = [[0]*3, [0]*3, [0]*3]

    # criterion = torch.nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()

    for i, (features, labels, feature_paths) in tqdm(enumerate(dataloader)):
        target = torch.LongTensor([tag_to_ix[t[0]] for t in labels])

        feat = Variable(features.squeeze())
        # target = Variable(label)
        if config.use_gpu:
            feat = feat.cuda()
            # target = target.cuda()

        neg_log_likelihood = model.neg_log_likelihood(feat, target)
        loss_meter.add(neg_log_likelihood.data[0])
        result = model(feat)
        for i, j in zip(target, result[1]):
            val_cm[i][j] += 1

    model.train()
    val_accuracy = 100. * sum([val_cm[c][c] for c in range(config.num_classes)]) / np.sum(val_cm)

    return val_cm, val_accuracy, loss_meter.value()[0]


def test(**kwargs):
    config.parse(kwargs)

    # prepare data
    test_data = Feature_Dataset(config.data_root, config.test_paths, phase='test')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=config.num_workers)
    print('Test Feature Lists:', test_data.__len__())

    tag_to_ix = {"Z": 0, "C": 1, "R": 2, START_TAG: 3, STOP_TAG: 4}

    # prepare model
    model = BiLSTM_CRF(tag_to_ix=tag_to_ix, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)

    if config.load_model_path:
        model.load(config.load_model_path)
    if config.use_gpu:
        model.cuda()
    model.eval()

    # metric
    test_cm = [[0] * 3, [0] * 3, [0] * 3]

    # go through the model
    for i, (features, labels, feature_paths) in tqdm(enumerate(test_dataloader)):
        # prepare input
        target = torch.LongTensor([tag_to_ix[t[0]] for t in labels])

        feat = Variable(features.squeeze())
        # target = Variable(target)
        if config.use_gpu:
            feat = feat.cuda()
            # target = target.cuda()

        result = model(feat)  # (score, predict list)

        for t, r in zip(target, result[1]):
            test_cm[t][r] += 1

    SE, SP, ACC = calculate_index(test_cm)

    print('confusion matrix:')
    print(test_cm)
    print('Sensitivity:', SE)
    print('Specificity:', SP)
    print('test accuracy:', ACC)


def test_output(**kwargs):
    config.parse(kwargs)

    # prepare data
    test_data = Feature_Dataset(config.data_root, config.test_paths, phase='test_output')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=config.num_workers)
    print('Test Feature Lists:', test_data.__len__())

    tag_to_ix = {"Z": 0, "C": 1, "R": 2, START_TAG: 3, STOP_TAG: 4}

    # prepare model
    model = BiLSTM_CRF(tag_to_ix=tag_to_ix, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)

    if config.load_model_path:
        model.load(config.load_model_path)
    if config.use_gpu:
        model.cuda()
    model.eval()

    misclassified, results = [], []

    # go through the model
    for i, (features, labels, feature_paths) in tqdm(enumerate(test_dataloader)):
        # prepare input
        target = []
        for t in labels:
            if t[0] != "H":
                target.append(tag_to_ix[t[0]])
            else:
                target.append(3)
        target = torch.LongTensor(target)

        feat = Variable(features.squeeze())
        # target = Variable(target)
        if config.use_gpu:
            feat = feat.cuda()
            # target = target.cuda()

        result = model(feat)

        for path, predict, true_label in zip(feature_paths, result[1], target):
            if int(predict) == 2:
                predict = 3
            if int(true_label) == 2:
                true_label = 3
            elif int(true_label) == 3:
                true_label = 2
            results.append((path[0].split('Data')[1], int(predict), int(true_label), 0, 0, 0))  # 三个0是为了和之前保持一致，填位置
            if predict != int(true_label):
                misclassified.append((path[0].split('Data')[1], int(predict), int(true_label), 0, 0, 0))
    write_csv('results/' + config.results_file, ['image_path', 'predict', 'true_label', 'prob_1', 'prob_2', 'prob_3'], results)
    write_csv('results/' + config.misclassified_file, ['image_path', 'predict', 'true_label', 'prob_1', 'prob_2', 'prob_3'], misclassified)


if __name__ == '__main__':
    fire.Fire()
