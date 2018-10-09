import itertools
import csv
import numpy as np
import matplotlib.pyplot as plt


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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.get_cmap('Reds'),
                          savname='Confusion Matrix.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(savname)


if __name__ == '__main__':
    # cm = np.array([[26112, 2208, 2348], [141, 1028, 31], [141, 27, 808]])
    # cm = np.array([[26339, 1868, 2461], [142, 1017, 41], [176, 18, 782]])
    cm = np.array([[233, 25, 29], [0, 71, 2], [3, 3, 59]])

    plot_confusion_matrix(cm=cm,
                          classes=['JianKang', 'ChengGu', 'RongGu'],
                          normalize=True,
                          savname='../results/Lesion-DenseNetSUP.png')


