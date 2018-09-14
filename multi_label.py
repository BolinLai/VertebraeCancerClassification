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
from dataset import MultiLabel_Dataset
from models import MultiTask_DenseNet121, CheXPre_MultiTask_DenseNet121
from utils import Visualizer, write_csv, calculate_index


def multitask_train(**kwargs):
    config.parse(kwargs)
    vis = Visualizer(port=2333, env=config.env)

    # prepare data
    train_data = MultiLabel_Dataset(config.data_root, config.train_paths, phase='train', balance=config.data_balance)
    val_data = MultiLabel_Dataset(config.data_root, config.test_paths, phase='val', balance=config.data_balance)
    print('Training Images:', train_data.__len__(), 'Validation Images:', val_data.__len__())

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # prepare model
    # model = MultiTask_DenseNet121(num_classes=2)  # 每一个分支都是2分类
    model = CheXPre_MultiTask_DenseNet121(num_classes=2)  # 每一个分支都是2分类

    if config.load_model_path:
        model.load(config.load_model_path)
    if config.use_gpu:
        model.cuda()

    model.train()

    # criterion and optimizer
    # F, T1, T2 = 3500, 3078, 3565  # 权重，分别是没病，成骨型，溶骨型的图片数量
    # weight_1 = torch.FloatTensor([T1/(F+T1+T2), (F+T2)/(F+T1+T2)]).cuda()  # weight也需要用cuda的
    # weight_2 = torch.FloatTensor([T2/(F+T1+T2), (F+T1)/(F+T1+T2)]).cuda()
    # criterion_1 = torch.nn.CrossEntropyLoss(weight=weight_1)
    # criterion_2 = torch.nn.CrossEntropyLoss(weight=weight_2)
    criterion_1 = torch.nn.CrossEntropyLoss()
    criterion_2 = torch.nn.CrossEntropyLoss()
    lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    # metrics
    softmax = functional.softmax
    loss_meter_1 = meter.AverageValueMeter()
    loss_meter_2 = meter.AverageValueMeter()
    loss_meter_total = meter.AverageValueMeter()
    train_cm_1 = meter.ConfusionMeter(2)  # 每个支路都是二分类，整体是三分类
    train_cm_2 = meter.ConfusionMeter(2)
    train_cm_total = meter.ConfusionMeter(3)
    previous_loss = 100
    previous_acc = 0

    # train
    if not os.path.exists(os.path.join('checkpoints', model.model_name)):
        os.mkdir(os.path.join('checkpoints', model.model_name))

    for epoch in range(config.max_epoch):
        loss_meter_1.reset()
        loss_meter_2.reset()
        loss_meter_total.reset()
        train_cm_1.reset()
        train_cm_2.reset()
        train_cm_total.reset()

        # train
        for i, (image, label_1, label_2, label, image_path) in tqdm(enumerate(train_dataloader)):
            # prepare input
            img = Variable(image)
            target_1 = Variable(label_1)
            target_2 = Variable(label_2)
            target = Variable(label)
            if config.use_gpu:
                img = img.cuda()
                target_1 = target_1.cuda()
                target_2 = target_2.cuda()
                target = target.cuda()

            # go through the model
            score_1, score_2 = model(img)

            # backpropagate
            optimizer.zero_grad()
            loss_1 = criterion_1(score_1, target_1)
            loss_2 = criterion_2(score_2, target_2)
            loss = loss_1 + loss_2
            # loss.backward()
            # optimizer.step()
            loss_1.backward(retain_graph=True)  # 这里将两个loss相加后回传的效果不太好，反而是分别回传效果更好
            optimizer.step()                    # 可能的原因是分别回传时的momentum算了两次，更容易突破局部最优解
            loss_2.backward()
            optimizer.step()

            # calculate loss and confusion matrix
            loss_meter_1.add(loss_1.data[0])
            loss_meter_2.add(loss_2.data[0])
            loss_meter_total.add(loss.data[0])

            p_1, p_2 = softmax(score_1, dim=1), softmax(score_2, dim=1)
            c = []

            # -----------------------------------------------------------------------
            for j in range(p_1.data.size()[0]):  # 将两个支路合并得到最终的预测结果
                if p_1.data[j][1] < 0.5 and p_2.data[j][1] < 0.5:
                    c.append([1, 0, 0])
                else:
                    if p_1.data[j][1] > p_2.data[j][1]:
                        c.append([0, 1, 0])
                    else:
                        c.append([0, 0, 1])
            # -----------------------------------------------------------------------

            train_cm_1.add(p_1.data, target_1.data)
            train_cm_2.add(p_2.data, target_2.data)
            train_cm_total.add(torch.FloatTensor(c), target.data)

            if i % config.print_freq == config.print_freq - 1:
                vis.plot_many({'loss_1': loss_meter_1.value()[0], 'loss_2': loss_meter_2.value()[0], 'loss_total': loss_meter_total.value()[0]})
                print('loss_1:', loss_meter_1.value()[0], 'loss_2:', loss_meter_2.value()[0], 'loss_total:', loss_meter_total.value()[0])

        # print result
        train_accuracy_1 = 100. * sum([train_cm_1.value()[c][c] for c in range(2)]) / train_cm_1.value().sum()
        train_accuracy_2 = 100. * sum([train_cm_2.value()[c][c] for c in range(2)]) / train_cm_2.value().sum()
        train_accuracy_total = 100. * sum([train_cm_total.value()[c][c] for c in range(3)]) / train_cm_total.value().sum()

        val_cm_1, val_accuracy_1, val_loss_1, val_cm_2, val_accuracy_2, val_loss_2, val_cm_total, val_accuracy_total, val_loss_total = multitask_val(model, val_dataloader)

        if val_accuracy_total > previous_acc:
            if config.save_model_name:
                model.save(os.path.join('checkpoints', model.model_name, config.save_model_name))
            else:
                model.save(os.path.join('checkpoints', model.model_name, model.model_name+'_best_model.pth'))
            previous_acc = val_accuracy_total

        vis.plot_many({'train_accuracy_1': train_accuracy_1, 'val_accuracy_1': val_accuracy_1,
                       'train_accuracy_2': train_accuracy_2, 'val_accuracy_2': val_accuracy_2,
                       'total_train_accuracy': train_accuracy_total, 'total_val_accuracy': val_accuracy_total})
        vis.log("epoch: [{epoch}/{total_epoch}], lr: {lr}, loss_1: {loss_1}, loss_2: {loss_2}, loss_total: {loss_total}".format(
            epoch=epoch + 1, total_epoch=config.max_epoch, lr=lr, loss_1=loss_meter_1.value()[0], loss_2=loss_meter_2.value()[0], loss_total=loss_meter_total.value()[0]))
        vis.log('train_cm_1:' + str(train_cm_1.value()) + ' train_cm_2:' + str(train_cm_2.value()) + ' train_cm_total:' + str(train_cm_total.value()))
        vis.log('val_cm_1:' + str(val_cm_1.value()) + ' val_cm_2:' + str(val_cm_2.value()) + ' val_cm_total:' + str(val_cm_total.value()))

        print('train_accuracy_1:', train_accuracy_1, 'val_accuracy_1:', val_accuracy_1,
              'train_accuracy_2:', train_accuracy_2, 'val_accuracy_2:', val_accuracy_2,
              'total_train_accuracy:', train_accuracy_total, 'total_val_accuracy:', val_accuracy_total)
        print("epoch: [{epoch}/{total_epoch}], lr: {lr}, loss_1: {loss_1}, loss_2: {loss_2}, loss_total: {loss_total}".format(
            epoch=epoch + 1, total_epoch=config.max_epoch, lr=lr, loss_1=loss_meter_1.value()[0], loss_2=loss_meter_2.value()[0], loss_total=loss_meter_total.value()[0]))
        print('train_cm_1:\n' + str(train_cm_1.value()) + '\ntrain_cm_2:\n' + str(train_cm_2.value()) + '\ntrain_cm_total:\n' + str(train_cm_total.value()))
        print('val_cm_1:\n' + str(val_cm_1.value()) + '\nval_cm_2:\n' + str(val_cm_2.value()) + '\nval_cm_total:\n' + str(val_cm_total.value()))

        # update learning rate
        if loss_meter_total.value()[0] > previous_loss:  # 可以考虑分别用两支的loss来判断
            lr = lr * config.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter_total.value()[0]


def multitask_val(model, dataloader):
    model.eval()
    val_cm_1 = meter.ConfusionMeter(2)
    val_cm_2 = meter.ConfusionMeter(2)
    val_cm_total = meter.ConfusionMeter(3)
    softmax = functional.softmax

    criterion_1 = torch.nn.CrossEntropyLoss()
    criterion_2 = torch.nn.CrossEntropyLoss()
    loss_meter_1 = meter.AverageValueMeter()
    loss_meter_2 = meter.AverageValueMeter()
    loss_meter_total = meter.AverageValueMeter()

    for i, (image, label_1, label_2, label, image_path) in tqdm(enumerate(dataloader)):
        img = Variable(image, volatile=True)
        target_1 = Variable(label_1)
        target_2 = Variable(label_2)
        target = Variable(label)
        if config.use_gpu:
            img = img.cuda()
            target_1 = target_1.cuda()
            target_2 = target_2.cuda()
            target = target.cuda()

        # go through the model
        score_1, score_2 = model(img)

        # calculate loss and confusion matrix
        loss_1 = criterion_1(score_1, target_1)
        loss_2 = criterion_2(score_2, target_2)
        loss = loss_1 + loss_2
        loss_meter_1.add(loss_1.data[0])
        loss_meter_2.add(loss_2.data[0])
        loss_meter_total.add(loss.data[0])

        p_1, p_2 = softmax(score_1, dim=1), softmax(score_2, dim=1)
        c = []

        # -----------------------------------------------------------------------
        for j in range(p_1.data.size()[0]):  # 将两个支路合并得到最终的预测结果
            if p_1.data[j][1] < 0.5 and p_2.data[j][1] < 0.5:
                c.append([1, 0, 0])
            else:
                if p_1.data[j][1] > p_2.data[j][1]:
                    c.append([0, 1, 0])
                else:
                    c.append([0, 0, 1])
        # -----------------------------------------------------------------------

        val_cm_1.add(p_1.data, target_1.data)
        val_cm_2.add(p_2.data, target_2.data)
        val_cm_total.add(torch.FloatTensor(c), target.data)

    model.train()
    val_accuracy_1 = 100. * sum([val_cm_1.value()[c][c] for c in range(2)]) / val_cm_1.value().sum()
    val_accuracy_2 = 100. * sum([val_cm_2.value()[c][c] for c in range(2)]) / val_cm_2.value().sum()
    val_accuracy_total = 100. * sum([val_cm_total.value()[c][c] for c in range(3)]) / val_cm_total.value().sum()

    return val_cm_1, val_accuracy_1, loss_meter_1.value()[0], \
           val_cm_2, val_accuracy_2, loss_meter_2.value()[0], \
           val_cm_total, val_accuracy_total, loss_meter_total.value()[0]


def multitask_test(**kwargs):
    config.parse(kwargs)

    # prepare data
    test_data = MultiLabel_Dataset(config.data_root, config.test_paths, phase='test')  # 注意这里不要加balance=False，否则生成的Dataset会包含混合型
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    print('Test Images:', test_data.__len__())

    # prepare model
    # model = MultiTask_DenseNet121(num_classes=2)
    model = CheXPre_MultiTask_DenseNet121(num_classes=2)

    if config.load_model_path:
        model.load(config.load_model_path)
    if config.use_gpu:
        model.cuda()
    model.eval()

    test_cm_1 = meter.ConfusionMeter(2)
    test_cm_2 = meter.ConfusionMeter(2)
    test_cm_total = meter.ConfusionMeter(3)
    softmax = functional.softmax

    # misclassified = []

    # go through the model
    for i, (image, label_1, label_2, label, image_path) in tqdm(enumerate(test_dataloader)):
        img = Variable(image, volatile=True)
        target_1 = Variable(label_1)
        target_2 = Variable(label_2)
        target = Variable(label)
        if config.use_gpu:
            img = img.cuda()
            target_1 = target_1.cuda()
            target_2 = target_2.cuda()
            target = target.cuda()

        # go through the model
        score_1, score_2 = model(img)

        p_1, p_2 = softmax(score_1, dim=1), softmax(score_2, dim=1)
        c = []

        # -----------------------------------------------------------------------
        for j in range(p_1.data.size()[0]):  # 将两个支路合并得到最终的预测结果
            if p_1.data[j][1] < 0.5 and p_2.data[j][1] < 0.5:
                c.append([1, 0, 0])
            else:
                if p_1.data[j][1] > p_2.data[j][1]:
                    c.append([0, 1, 0])
                else:
                    c.append([0, 0, 1])
        # -----------------------------------------------------------------------

        # misclassified image path
        # for path, predicted, true_label in zip(image_path, c, target):
        #     if np.argmax(predicted, axis=0) != int(true_label):
        #         misclassified.append((path, np.argmax(predicted, axis=0), int(true_label)))
        # misclassified.extend([(path, np.argmax(predicted, axis=0), int(true_label)) for path, predicted, true_label in zip(image_path, c, target) if np.argmax(predicted, axis=0) != int(true_label)])

        test_cm_1.add(p_1.data, target_1.data)
        test_cm_2.add(p_2.data, target_2.data)
        test_cm_total.add(torch.FloatTensor(c), target.data)

    test_accuracy_1 = 100. * sum([test_cm_1.value()[c][c] for c in range(2)]) / test_cm_1.value().sum()
    test_accuracy_2 = 100. * sum([test_cm_2.value()[c][c] for c in range(2)]) / test_cm_2.value().sum()
    # test_accuracy_total = 100. * sum([test_cm_total.value()[c][c] for c in range(3)]) / test_cm_total.value().sum()
    SE, SP, ACC = calculate_index(test_cm_total.value())

    print('test_cm_1:\n', test_cm_1.value(), '\ntest_cm_2:\n', test_cm_2.value(), '\ntest_cm_total:\n', test_cm_total.value())
    print('test_accuracy_1:', test_accuracy_1, 'test_accuracy_2:', test_accuracy_2, 'test_accuracy_total:', ACC)
    print('total sensitivity:', SE)
    print('total specificity:', SP)


def multitask_test_output(**kwargs):
    config.parse(kwargs)

    # prepare data
    test_data = MultiLabel_Dataset(config.data_root, config.test_paths, phase='test_output')  # 注意这里不要加balance=False，否则生成的Dataset会包含混合型
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    print('Test Image:', test_data.__len__())

    # prepare model
    # model = MultiTask_DenseNet121(num_classes=2)
    model = CheXPre_MultiTask_DenseNet121(num_classes=2)

    if config.load_model_path:
        model.load(config.load_model_path)
    if config.use_gpu:
        model.cuda()
    model.eval()

    softmax = functional.softmax
    misclassified, results = [], []

    # go through the model
    for i, (image, label_1, label_2, label, image_path) in tqdm(enumerate(test_dataloader)):
        img = Variable(image, volatile=True)
        target_1 = Variable(label_1)
        target_2 = Variable(label_2)
        target = Variable(label)
        if config.use_gpu:
            img = img.cuda()
            target_1 = target_1.cuda()
            target_2 = target_2.cuda()
            target = target.cuda()

        score_1, score_2 = model(img)

        p_1, p_2 = softmax(score_1, dim=1), softmax(score_2, dim=1)
        c = []

        # -----------------------------------------------------------------------
        for j in range(p_1.data.size()[0]):  # 将两个支路合并得到最终的预测结果
            if p_1.data[j][1] < 0.5 and p_2.data[j][1] < 0.5:
                c.append([1, 0, 0])
            else:
                if p_1.data[j][1] > p_2.data[j][1]:
                    c.append([0, 1, 0])
                else:
                    c.append([0, 0, 1])
        # -----------------------------------------------------------------------

        # collect results of each slice
        for path, predicted, true_label, prob in zip(image_path, np.argmax(torch.FloatTensor(c), 1), target, torch.FloatTensor(c)):
            if int(predicted) == 2:  # 将预测的成骨型的类别从2变为3，保证csv文件中真实label和预测labe的一致性
                predicted = 3
            results.append((path.split('patient_image_4')[1], int(predicted), int(true_label), prob[0], prob[1], prob[2]))
            if predicted != int(true_label):
                misclassified.append((path.split('patient_image_4')[1], int(predicted), int(true_label), prob[0], prob[1], prob[2]))

        # results.extend([(path.split('patient_image_4')[1], int(predicted), int(true_label), prob[0], prob[1], prob[2])
        #                 for path, predicted, true_label, prob in zip(image_path, np.argmax(softmax(score, dim=1).data, 1), target, softmax(score, dim=1).data)])
        # misclassified.extend([(path.split('patient_image_4')[1], int(predicted), int(true_label), prob[0], prob[1], prob[2])
        #                       for path, predicted, true_label, prob in zip(image_path, np.argmax(softmax(score, dim=1).data, 1), target, softmax(score, dim=1).data) if predicted != int(true_label)])

    write_csv('results/'+config.results_file, ['image_path', 'predict', 'true_label', 'prob_1', 'prob_2', 'prob_3'], results)
    write_csv('results/'+config.misclassified_file, ['image_path', 'predict', 'true_label', 'prob_1', 'prob_2', 'prob_3'], misclassified)


if __name__ == '__main__':
    fire.Fire()
