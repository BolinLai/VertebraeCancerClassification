# coding: utf-8
import fire
import csv
import numpy as np

from tqdm import tqdm
from pprint import pprint


def write_csv(file, tag, content):
    """

    :param file:
    :param tag: A list of names of per coloumn
    :param content:
    :return:
    """
    with open(file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(tag)
        writer.writerows(content)


def calculate_index(matrix):
    """
    计算灵敏度、特异度等一系列指标

    :param matrix: 混淆矩阵
    :return:
    """
    if len(matrix) != len(matrix[0]):
        raise IndexError

    num_class = len(matrix)
    SE, SP = [], []

    for c in range(num_class):
        SE.append(matrix[c][c] / sum(matrix[c]))  # 第c类的灵敏度为第c行第c列的数除以第c行的和

        sum_row = sum(matrix[c])
        sum_column = sum([matrix[i][j] for i in range(num_class) for j in range(num_class) if j == c])
        # 第c类的特异度为混淆矩阵中，除了第c行及第c列之外元素的和，除以矩阵中除了第c行之外元素的和
        SP.append((np.sum(matrix) - sum_row - sum_column + matrix[c][c]) / (np.sum(matrix) - sum_row))
    ACC = sum([matrix[i][i] for i in range(num_class)]) / np.sum(matrix)
    return SE, SP, ACC


def lesion_localize(results, classes, changelabel):
    patient_id = results[0].strip().split(',')[0].split('/')[2]  # 第一张image的病人id
    lesion_label = int(results[0].strip().split(',')[2])  # 第一张image的label
    lesions = []  # 记录病灶的起始index
    start = 0

    for i, x in enumerate(results):
        # print(x.strip().split(','))
        patient = x.strip().split(',')[0].split('/')[2]
        label = int(x.strip().split(',')[2])
        # print(patient, predicted, label)

        if changelabel:
            if classes == 3:
                if label == 2:  # 交换label，方便写混淆矩阵部分的代码
                    label = 3
                elif label == 3:
                    label = 2
                else:
                    pass
            else:
                pass

        if label != lesion_label or patient != patient_id:
            lesions.append((start, i - 1, lesion_label))
            start = i
            patient_id = patient
            lesion_label = label
    return lesions


def predict_lesion_localize(results, classes):
    patient_id = results[0].strip().split(',')[0].split('/')[2]  # 第一张image的病人id
    lesion_predict = int(results[0].strip().split(',')[1])  # 第一张image的预测结果
    lesions = []  # 记录病灶的起始index
    start = 0

    for i, x in enumerate(results):
        # print(x.strip().split(','))
        patient = x.strip().split(',')[0].split('/')[2]
        predict = int(x.strip().split(',')[1])
        # print(patient, predicted, label)

        # if classes == 3:
        #     if label == 2:  # 交换label，方便写混淆矩阵部分的代码
        #         label = 3
        #     elif label == 3:
        #         label = 2
        #     else:
        #         pass
        # else:
        #     pass

        if predict != lesion_predict or patient != patient_id:
            lesions.append((start, i - 1, lesion_predict))
            start = i
            patient_id = patient
            lesion_predict = predict
    return lesions


def label_based_3class(results_file):
    """
    以标注为基准定位病灶，计算准确率，混合型不加入统计。
    """
    def lesion_predict(dic):
        total = dic['0'] + dic['1'] + dic['2']
        if dic['0'] / total > 0.7:
            re = 0
        elif dic['1'] > dic['2']:
            re = 1
        else:
            re = 2
        return re

    with open(results_file, 'r') as f:
        results = f.readlines()[1:]

    lesions = lesion_localize(results, classes=3, changelabel=True)  # 每个病灶的起始index，并且交换2和3的标签
    # lesions.sort(key=lambda m: m[1]-m[0], reverse=False)
    print('total lesions:', len(lesions))
    # pprint(lesions)

    result_dict = {'0': 0, '1': 0, '2': 0}
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for idx in tqdm(lesions):
        lesion_results = results[idx[0]: idx[1]+1]  # 这里要有+1，因为list截取一段不包含最后一个index

        # # 抛弃每段病灶的起始和结束的过度部分
        # length = idx[1]-idx[0]+1
        # if length >= 10:
        #     lesion_results = results[int(idx[0] + length*0.2): int(idx[1]+1 - length*0.2)]
        # elif 5 < length < 10:
        #     lesion_results = results[idx[0]+1: idx[1]]
        # else:
        #     lesion_results = results[idx[0]: idx[1]+1]

        lesion_label = idx[2]

        for x in lesion_results:
            predicted = int(x.strip().split(',')[1])
            if predicted == 3:  # 转换csv中预测的结果
                predicted = 2
            result_dict[str(predicted)] += 1
        # print(idx, result_dict)
        lesion_predicted = lesion_predict(result_dict)

        if lesion_label != 3:  # 对混合型不计入混淆矩阵
            cm[lesion_label][lesion_predicted] += 1

        result_dict = {'0': 0, '1': 0, '2': 0}
    SE, SP, ACC = calculate_index(cm)
    print(cm[0], '\n', cm[1], '\n', cm[2])
    print('SE_1:', SE[1])
    print('SE_2:', SE[2])
    print('SP_1:', SP[1])
    print('SP_2:', SP[2])
    print('ACC:', ACC)


def label_based_4class(results_file):
    """
    以标注为基准定位病灶，计算准确率，统计混合型。
    """
    def lesion_predict(dic):
        total = dic['0'] + dic['1'] + dic['2']
        pos_total = dic['1'] + dic['2']
        if dic['0'] / total > 0.7:
            re = 0
        elif dic['1'] / pos_total > 0.7:
            re = 1
        elif dic['2'] / pos_total > 0.7:
            re = 3
        else:
            re = 2  # 当成骨和溶骨占比均小于0.7时记为混合型
        return re
    with open(results_file, 'r') as f:
        results = f.readlines()[1:]

    lesions = lesion_localize(results, classes=4, changelabel=False)  # 每个病灶的起始index
    # lesions.sort(key=lambda m: m[1]-m[0], reverse=False)
    print('total lesions:', len(lesions))
    # pprint(lesions)

    result_dict = {'0': 0, '1': 0, '2': 0}
    cm = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    for idx in tqdm(lesions):
        lesion_results = results[idx[0]: idx[1] + 1]  # 这里要有+1，因为list截取一段不包含最后一个index

        # # 抛弃每段病灶的起始和结束的过度部分
        # length = idx[1]-idx[0]+1
        # if length >= 10:
        #     lesion_results = results[int(idx[0] + length*0.2): int(idx[1]+1 - length*0.2)]
        # elif 5 < length < 10:
        #     lesion_results = results[idx[0]+1: idx[1]]
        # else:
        #     lesion_results = results[idx[0]: idx[1]+1]

        lesion_label = idx[2]

        for x in lesion_results:
            predicted = int(x.strip().split(',')[1])
            if predicted == 3:  # 转换csv中预测的label
                predicted = 2
            result_dict[str(predicted)] += 1
        # print(idx, result_dict)
        lesion_predicted = lesion_predict(result_dict)

        cm[lesion_label][lesion_predicted] += 1

        result_dict = {'0': 0, '1': 0, '2': 0}
    print(cm[0], '\n', cm[1], '\n', cm[2], '\n', cm[3])
    print((cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3])/np.sum(cm))

    SE, SP, ACC = calculate_index(cm)
    print(cm[0], '\n', cm[1], '\n', cm[2])
    print('SE_1:', SE[1])
    print('SE_2:', SE[2])
    print('SE_3:', SE[3])
    print('SP_1:', SP[1])
    print('SP_2:', SP[2])
    print('SP_3:', SP[3])
    print('ACC:', ACC)


def predict_based_3class(results_file):
    with open(results_file, 'r') as f:
        results = f.readlines()[1:]

    # ============================ Step1:将预测结果按照阈值和判断方法截断 =================================

    len_threshold = 5

    truncated_results = []  # 将读入的结果按照不同病人的id，去除混合型和是否超过长度阈值的判断方法截断，便于后续操作
    temp = []
    for i, x in enumerate(results):
        image_path = x.strip().split(',')[0]
        predict = int(x.strip().split(',')[1])
        label = int(x.strip().split(',')[2])
        prob_1 = x.strip().split(',')[3]
        prob_2 = x.strip().split(',')[4]
        prob_3 = x.strip().split(',')[5]

        if not temp:  # 如果temp为空，即这是一段脊骨结果的第一张slice
            if label != 2:
                temp.append([image_path, predict, label, prob_1, prob_2, prob_3])
            else:
                pass
        else:
            if label != 2:
                patient_id = image_path.split('/')[2]
                prepatient_id = results[i-1].strip().split(',')[0].split('/')[2]
                prelabel = int(results[i-1].strip().split(',')[2])

                if patient_id == prepatient_id and prelabel != 2:
                    temp.append([image_path, predict, label, prob_1, prob_2, prob_3])
                else:  # 如果当前slice的病人id和上一张不同，或者上一张的label是2，则应该从这里截断
                    if len(temp) >= len_threshold:  # 对于本身长度就少于t的病灶直接舍弃
                        truncated_results.append(temp)
                    # truncated_results.append(temp)
                    temp = [[image_path, predict, label, prob_1, prob_2, prob_3]]
            else:
                pass

    # pprint(truncated_results[7])
    # print(len(truncated_results[7]))

    # ============================ Step2:将单张预测结果以病灶为单位进行修正 =================================

    processed_results = []

    for res in tqdm(truncated_results):
        # 对于每一段截断脊骨，判断第一张的label
        for i, x in enumerate(res):
            if i+len_threshold <= len(res):
                res_next = [res[i+j][1] for j in range(len_threshold)]
                # print(res_next)
                # print([x[1]] * len_threshold)
                if res_next == [x[1]] * len_threshold:
                    prepredict = x[1]
                    break
                else:
                    pass
            else:
                pass
        else:  # 如果for循环里没有执行过break，即没有连续t个相同的slice，取这一段slice中数量最多的predict
            d = {'0': 0, '1': 0, '3': 0}
            for x in res:
                d[str(x[1])] += 1
            prepredict = int(sorted(d.items(), key=lambda y: y[1], reverse=True)[0][0])

        for i, x in enumerate(res):
            if x[1] == prepredict:  # 如果当前slice的predict和上一张一致，则直接加入
                processed_results.append(x[:3])
            else:
                if i+len_threshold > len(res):  # 如果当前slice的predict和上一张不一致，但是剩余的slice数不足t张，则修改为和之前一致
                    processed_results.append([x[0], prepredict, x[2]])
                else:
                    res_next = [res[i+j][1] for j in range(len_threshold)]
                    if res_next == [x[1]] * len_threshold:  # 如果当前slice和predict和上一张不一致，且之后连续相同的超过t张，则直接加入并修改prepredict
                        processed_results.append(x[:3])
                        prepredict = x[1]
                    else:  # 如果当前slice和predict和上一张不一致，且之后连续相同的不足t张，则修改为和之前一致
                        processed_results.append([x[0], prepredict, x[2]])

    write_csv('processed_results.csv', ['path', 'predict', 'label'], processed_results)

    # ============================== Step3:由单张结果得到病灶结果并统计指标 ====================================

    with open('processed_results.csv', 'r') as f:
        processed_results = f.readlines()[1:]
    predict_lesions = predict_lesion_localize(processed_results, classes=3)
    label_lesions = lesion_localize(processed_results, classes=3, changelabel=False)
    # pprint(lesions[:10])
    # pprint(label_lesions[:10])

    IOU_threshold = 0.5
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for preles in tqdm(predict_lesions):
        prestart, preend, predict = preles[0], preles[1], preles[2]
        pre_slice_index = [x for x in range(prestart, preend+1)]

        label_slice_index = {'0': [], '1': [], '2': [], '3': []}
        for lables in label_lesions:
            labstart, labend, label = lables[0], lables[1], lables[2]
            if prestart > labend or preend < labstart:  # 如果无交集
                pass
            else:
                # print(labstart, labend, prestart, preend)
                for x in range(labstart, labend+1):
                    label_slice_index[str(label)].append(x)

        # print(len(label_slice_index['0']))
        IOU = [len(set(pre_slice_index) & set(label_slice_index['0'])) / len(set(pre_slice_index) | set(label_slice_index['0'])),
               len(set(pre_slice_index) & set(label_slice_index['1'])) / len(set(pre_slice_index) | set(label_slice_index['1'])),
               len(set(pre_slice_index) & set(label_slice_index['2'])) / len(set(pre_slice_index) | set(label_slice_index['2'])),
               len(set(pre_slice_index) & set(label_slice_index['3'])) / len(set(pre_slice_index) | set(label_slice_index['3']))]
        # print(set(pre_slice_index))
        # print(set(label_slice_index['0']))
        # print(len(set(pre_slice_index) & set(label_slice_index['0'])))
        # print("====================")

        # true_label = int(np.argmax(IOU[1:])) + 1  # 如果三类病中IOU最大的超过阈值则以此为label，否则label为0
        # if IOU[true_label] < IOU_threshold:
        #     true_label = 0
        if IOU[predict] > IOU_threshold:
            true_label = predict
        else:
            true_label = int(np.argmax(IOU))

        if true_label == 2:
            true_label = 3
        elif true_label == 3:
            true_label = 2
        else:
            # print(2)
            pass

        if predict == 3:
            predict = 2

        if true_label != 3:
            cm[true_label][predict] += 1

    SE, SP, ACC = calculate_index(cm)

    print('confusion matrix')
    print(cm[0], '\n', cm[1], '\n', cm[2])
    print('SE_1:', SE[1])
    print('SE_2:', SE[2])
    print('SP_1:', SP[1])
    print('SP_2:', SP[2])
    print('ACC:', ACC)


def predict_based_4class(results_file):
    """
    以预测的结果为基准定位病灶，计算准确率，统计混合型。
    """
    with open(results_file, 'r') as f:
        results = f.readlines()[1:]

    # ============================ Step1:将预测结果按照阈值和判断方法截断 =================================

    len_threshold = 5

    truncated_results = []  # 将读入的结果按照不同病人的id，是否超过长度阈值的判断方法截断，便于后续操作
    temp = []
    for i, x in enumerate(results):
        image_path = x.strip().split(',')[0]
        predict = int(x.strip().split(',')[1])
        label = int(x.strip().split(',')[2])
        prob_1 = x.strip().split(',')[3]
        prob_2 = x.strip().split(',')[4]
        prob_3 = x.strip().split(',')[5]

        if not temp:  # 如果temp为空，即这是一段脊骨结果的第一张slice
            temp.append([image_path, predict, label, prob_1, prob_2, prob_3])
        else:
            patient_id = image_path.split('/')[2]
            prepatient_id = results[i-1].strip().split(',')[0].split('/')[2]

            if patient_id == prepatient_id:
                temp.append([image_path, predict, label, prob_1, prob_2, prob_3])
            else:  # 如果当前slice的病人id和上一张不同，则应该从这里截断
                if len(temp) >= len_threshold:  # 对于本身长度就少于t的病灶直接舍弃
                    truncated_results.append(temp)
                # truncated_results.append(temp)
                temp = [[image_path, predict, label, prob_1, prob_2, prob_3]]


    # pprint(truncated_results[7])
    # print(len(truncated_results[7]))

    # ============================ Step2:将单张预测结果以病灶为单位进行修正 =================================

    processed_results = []

    for res in tqdm(truncated_results):
        # 对于每一段截断脊骨，判断第一张的label
        for i, x in enumerate(res):
            if i+len_threshold <= len(res):
                res_next = [res[i+j][1] for j in range(len_threshold)]
                # print(res_next)
                # print([x[1]] * len_threshold)
                if res_next == [x[1]] * len_threshold:
                    prepredict = x[1]
                    break
                else:
                    pass
            else:
                pass
        else:  # 如果for循环里没有执行过break，即没有连续t个相同的slice，取这一段slice中数量最多的predict
            d = {'0': 0, '1': 0, '2': 0, '3': 0}
            for x in res:
                d[str(x[1])] += 1
            prepredict = int(sorted(d.items(), key=lambda y: y[1], reverse=True)[0][0])

        for i, x in enumerate(res):
            if x[1] == prepredict:  # 如果当前slice的predict和上一张一致，则直接加入
                processed_results.append(x[:3])
            else:
                if i+len_threshold > len(res):  # 如果当前slice的predict和上一张不一致，但是剩余的slice数不足t张，则修改为和之前一致
                    processed_results.append([x[0], prepredict, x[2]])
                else:
                    res_next = [res[i+j][1] for j in range(len_threshold)]
                    if res_next == [x[1]] * len_threshold:  # 如果当前slice和predict和上一张不一致，且之后连续相同的超过t张，则直接加入并修改prepredict
                        processed_results.append(x[:3])
                        prepredict = x[1]
                    else:  # 如果当前slice和predict和上一张不一致，且之后连续相同的不足t张，则修改为和之前一致
                        processed_results.append([x[0], prepredict, x[2]])

    write_csv('results/processed_results.csv', ['path', 'predict', 'label'], processed_results)

    # ============================== Step3:由单张结果得到病灶结果并统计指标 ====================================

    with open('results/processed_results.csv', 'r') as f:
        processed_results = f.readlines()[1:]
    predict_lesions = predict_lesion_localize(processed_results, classes=4)
    label_lesions = lesion_localize(processed_results, classes=4, changelabel=False)
    # pprint(lesions[:10])
    # pprint(label_lesions[:10])

    IOU_threshold = 0.5
    cm = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    for preles in tqdm(predict_lesions):
        prestart, preend, predict = preles[0], preles[1], preles[2]
        pre_slice_index = [x for x in range(prestart, preend+1)]

        label_slice_index = {'0': [], '1': [], '2': [], '3': []}
        for lables in label_lesions:
            labstart, labend, label = lables[0], lables[1], lables[2]
            if prestart > labend or preend < labstart:  # 如果无交集
                pass
            else:
                # print(labstart, labend, prestart, preend)
                for x in range(labstart, labend+1):
                    label_slice_index[str(label)].append(x)

        # print(len(label_slice_index['0']))
        IOU = [len(set(pre_slice_index) & set(label_slice_index['0'])) / len(set(pre_slice_index) | set(label_slice_index['0'])),
               len(set(pre_slice_index) & set(label_slice_index['1'])) / len(set(pre_slice_index) | set(label_slice_index['1'])),
               len(set(pre_slice_index) & set(label_slice_index['2'])) / len(set(pre_slice_index) | set(label_slice_index['2'])),
               len(set(pre_slice_index) & set(label_slice_index['3'])) / len(set(pre_slice_index) | set(label_slice_index['3']))]
        # print(set(pre_slice_index))
        # print(set(label_slice_index['0']))
        # print(len(set(pre_slice_index) & set(label_slice_index['0'])))
        # print("====================")

        # true_label = int(np.argmax(IOU[1:])) + 1  # 如果三类病中IOU最大的超过阈值则以此为label，否则label为0
        # if IOU[true_label] < IOU_threshold:
        #     true_label = 0
        if IOU[predict] > IOU_threshold:
            true_label = predict
        else:
            true_label = int(np.argmax(IOU))

        cm[true_label][predict] += 1

    SE, SP, ACC = calculate_index(cm)

    print('confusion matrix')
    print(cm[0], '\n', cm[1], '\n', cm[2], '\n', cm[3])
    print('SE_1:', SE[1])
    print('SE_2:', SE[2])
    print('SE_3:', SE[3])
    print('SP_1:', SP[1])
    print('SP_2:', SP[2])
    print('SP_3:', SP[3])
    print('ACC:', ACC)
