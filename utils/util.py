import csv
import numpy as np


def write_csv(file, tag, content):
    """
    写入csv文件

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
