# coding: utf-8
import os
import fire
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional
from torchnet import meter

from config import config
from dataset import Vertebrae_Dataset, FrameDiff_Dataset
from models import ResNet34, DenseNet121, CheXPre_DenseNet121
from utils import Visualizer, write_csv, calculate_index


def train(**kwargs):
    config.parse(kwargs)
    vis = Visualizer(port=2333, env=config.env)

    # prepare data
    train_data = Vertebrae_Dataset(config.data_root, config.train_paths, phase='train', balance=config.data_balance)
    val_data = Vertebrae_Dataset(config.data_root, config.test_paths, phase='val', balance=config.data_balance)
    # train_data = FrameDiff_Dataset(config.data_root, config.train_paths, phase='train', balance=config.data_balance)
    # val_data = FrameDiff_Dataset(config.data_root, config.test_paths, phase='val', balance=config.data_balance)
    print('Training Images:', train_data.__len__(), 'Validation Images:', val_data.__len__())

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # prepare model
    # model = ResNet34(num_classes=4)
    model = DenseNet121(num_classes=config.num_classes)
    # model = CheXPre_DenseNet121(num_classes=config.num_classes)

    if config.load_model_path:
        model.load(config.load_model_path)
    if config.use_gpu:
        model.cuda()

    model.train()

    # criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    # metric
    softmax = functional.softmax
    loss_meter = meter.AverageValueMeter()
    train_cm = meter.ConfusionMeter(config.num_classes)
    previous_loss = 100
    previous_acc = 0

    # train
    if not os.path.exists(os.path.join('checkpoints', model.model_name)):
        os.mkdir(os.path.join('checkpoints', model.model_name))

    for epoch in range(config.max_epoch):
        loss_meter.reset()
        train_cm.reset()

        # train
        for i, (image, label, image_path) in tqdm(enumerate(train_dataloader)):
            # prepare input
            img = Variable(image)
            target = Variable(label)
            if config.use_gpu:
                img = img.cuda()
                target = target.cuda()

            # go through the model
            score = model(img)

            # backpropagate
            optimizer.zero_grad()
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.data[0])
            train_cm.add(softmax(score, dim=1).data, target.data)

            if i % config.print_freq == config.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])
                print('loss', loss_meter.value()[0])

        # print result
        train_accuracy = 100. * sum([train_cm.value()[c][c] for c in range(config.num_classes)]) / train_cm.value().sum()
        val_cm, val_accuracy, val_loss = val(model, val_dataloader)

        if val_accuracy > previous_acc:
            if config.save_model_name:
                model.save(os.path.join('checkpoints', model.model_name, config.save_model_name))
            else:
                model.save(os.path.join('checkpoints', model.model_name, model.model_name+'_best_model.pth'))
            previous_acc = val_accuracy

        vis.plot_many({'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy})
        vis.log("epoch: [{epoch}/{total_epoch}], lr: {lr}, loss: {loss}".format(
            epoch=epoch+1, total_epoch=config.max_epoch, lr=lr, loss=loss_meter.value()[0]))
        vis.log('train_cm:')
        vis.log(train_cm.value())
        vis.log('val_cm')
        vis.log(val_cm.value())
        print('train_accuracy:', train_accuracy, 'val_accuracy:', val_accuracy)
        print("epoch: [{epoch}/{total_epoch}], lr: {lr}, loss: {loss}".format(
            epoch=epoch+1, total_epoch=config.max_epoch, lr=lr, loss=loss_meter.value()[0]))
        print('train_cm:')
        print(train_cm.value())
        print('val_cm:')
        print(val_cm.value())

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * config.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]


def val(model, dataloader):
    model.eval()
    val_cm = meter.ConfusionMeter(config.num_classes)
    softmax = functional.softmax

    criterion = torch.nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()

    for i, (image, label, image_path) in tqdm(enumerate(dataloader)):
        img = Variable(image, volatile=True)
        target = Variable(label)
        if config.use_gpu:
            img = img.cuda()
            target = target.cuda()

        score = model(img)

        loss = criterion(score, target)
        loss_meter.add(loss.data[0])
        val_cm.add(softmax(score, dim=1).data, target.data)

    model.train()
    val_accuracy = 100. * sum([val_cm.value()[c][c] for c in range(config.num_classes)]) / val_cm.value().sum()

    return val_cm, val_accuracy, loss_meter.value()[0]


def test(**kwargs):
    config.parse(kwargs)

    # prepare data
    test_data = Vertebrae_Dataset(config.data_root, config.test_paths, phase='test')  # 注意这里不要加balance=False，否则生成的Dataset会包含混合型
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # test_data = FrameDiff_Dataset(config.data_root, config.test_paths, phase='test', balance=config.data_balance)
    # test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    print('Test Image:', test_data.__len__())

    # prepare model
    # model = ResNet34(num_classes=config.num_classes)
    model = DenseNet121(num_classes=config.num_classes)
    # model = CheXPre_DenseNet121(num_classes=config.num_classes)

    if config.load_model_path:
        model.load(config.load_model_path)
    if config.use_gpu:
        model.cuda()
    model.eval()

    test_cm = meter.ConfusionMeter(config.num_classes)
    softmax = functional.softmax

    # go through the model
    for i, (image, label, image_path) in tqdm(enumerate(test_dataloader)):
        img = Variable(image, volatile=True)
        target = Variable(label)
        if config.use_gpu:
            img = img.cuda()
            target = target.cuda()

        score = model(img)

        test_cm.add(softmax(score, dim=1).data, target.data)

        # collect results of each slice
        # for path, predicted, true_label, prob in zip(image_path, np.argmax(softmax(score, dim=1).data, 1), target, softmax(score, dim=1).data):
        #     results.append((path.split('patient_image_4')[1], int(predicted), int(true_label), prob[0], prob[1], prob[2]))
        #     if predicted != int(true_label):
        #         misclassified.append((path.split('patient_image_4')[1], int(predicted), int(true_label), prob[0], prob[1], prob[2]))

    SE, SP, ACC = calculate_index(test_cm.value())

    print('confusion matrix:')
    print(test_cm.value())
    print('Sensitivity:', SE)
    print('Specificity:', SP)
    print('test accuracy:', ACC)


def test_output(**kwargs):
    config.parse(kwargs)

    # prepare data
    test_data = Vertebrae_Dataset(config.data_root, config.test_paths, phase='test_output')  # 注意这里不要加balance=False，否则生成的Dataset会包含混合型
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    # test_data = FrameDiff_Dataset(config.data_root, config.test_paths, phase='test_output')  # 注意这里不要加balance=False，否则生成的Dataset会包含混合型
    # test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    print('Test Image:', test_data.__len__())

    # prepare model
    # model = ResNet34(num_classes=config.num_classes)
    model = DenseNet121(num_classes=config.num_classes)
    # model = CheXPre_DenseNet121(num_classes=config.num_classes)

    if config.load_model_path:
        model.load(config.load_model_path)
    if config.use_gpu:
        model.cuda()
    model.eval()

    softmax = functional.softmax
    misclassified, results = [], []

    # go through the model
    for i, (image, label, image_path) in tqdm(enumerate(test_dataloader)):
        img = Variable(image, volatile=True)
        target = Variable(label)
        if config.use_gpu:
            img = img.cuda()
            target = target.cuda()

        score = model(img)

        # collect results of each slice
        for path, predicted, true_label, prob in zip(image_path, np.argmax(softmax(score, dim=1).data, 1), target, softmax(score, dim=1).data):
            if int(predicted) == 2:  # 将预测的成骨型的类别从2变为3，保证csv文件中真实label和预测labe的一致性
                predicted = 3
            results.append((path.split('patient_image_4')[1], int(predicted), int(true_label), prob[0], prob[1], prob[2]))
            if predicted != int(true_label):
                misclassified.append((path.split('patient_image_4')[1], int(predicted), int(true_label), prob[0], prob[1], prob[2]))

        # results.extend([(path.split('patient_image_4')[1], int(predicted), int(true_label), prob[0], prob[1], prob[2])
        #                 for path, predicted, true_label, prob in zip(image_path, np.argmax(softmax(score, dim=1).data, 1), target, softmax(score, dim=1).data)])
        # misclassified.extend([(path.split('patient_image_4')[1], int(predicted), int(true_label), prob[0], prob[1], prob[2])
        #                       for path, predicted, true_label, prob in zip(image_path, np.argmax(softmax(score, dim=1).data, 1), target, softmax(score, dim=1).data) if predicted != int(true_label)])

    write_csv(os.path.join('results', config.results_file), ['image_path', 'predict', 'true_label', 'prob_1', 'prob_2', 'prob_3'], results)
    write_csv(os.path.join('results', config.misclassified_file), ['image_path', 'predict', 'true_label', 'prob_1', 'prob_2', 'prob_3'], misclassified)


def save_features(**kwargs):
    config.parse(kwargs)

    # prepare data  # 需要先把BasicDataset里的shuffle和交换label注释掉，同时trans只进行到归一化，将后面变为ImageNet的分布的部分注释掉，前面图片的翻转旋转注释掉
    train_data = Vertebrae_Dataset(config.data_root, config.train_paths, phase='train', balance=False)
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    print('Training Images:', train_data.__len__())

    # test_data = Vertebrae_Dataset(config.data_root, config.test_paths, phase='test', balance=False)
    # test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    # print('Test Images:', test_data.__len__())

    # prepare model
    # model = ResNet34(num_classes=4)
    model = DenseNet121(num_classes=config.num_classes)
    # model = CheXPre_DenseNet121(num_classes=config.num_classes)

    if config.load_model_path:
        model.load(config.load_model_path)
    if config.use_gpu:
        model.cuda()
    model.eval()

    for image, label, image_path in tqdm(train_dataloader):
        img = Variable(image, volatile=True)
        target = Variable(label)
        if config.use_gpu:
            img = img.cuda()
            target = target.cuda()

        model.save_feature(img, target, image_path, feature_folder='/DATA5_DB8/data/bllai/Data/Features_Horizontal_Vertical')


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'test': test,
        'test_output': test_output,
        'save_features': save_features
    })
