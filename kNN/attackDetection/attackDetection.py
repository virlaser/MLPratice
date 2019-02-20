# coding : utf-8

from nltk import FreqDist
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def load_user_command(filename):
    """
    读取用户操作文件，生成用户操作频率和操作列表
    :param filename: 用户操作文件
    :return: （用户操作列表，用户操作命令去重后）
    """
    command_lists = []
    dist = []
    with open(filename) as f:
        i = 0
        x = []
        for line in f:
            line = line.strip('\n')
            x.append(line)
            dist.append(line)
            i += 1
            # 每 100 个命令形成一个操作序列
            if i == 100:
                command_lists.append(x)
                x = []
                i = 0
    # FreqDist中的键为单词，值为单词的出现总次数
    fdist = FreqDist(dist).keys()
    return command_lists, fdist


def get_command_feature(command_lists, dist):
    """
    操作命令向量化
    :param command_lists: 用户操作列表
    :param dist: 用户操作命令去重后
    :return:
    """
    command_feature = []
    for command_list in command_lists:
        v = [0]*len(dist)
        for i in range(0, len(dist)):
            if list(dist)[i] in command_list:
                v[i] += 1
        command_feature.append(v)
    return command_feature


def get_lable(filename, index):
    """
    得到用户操作对应的标签
    :param filename:
    :param index:
    :return:
    """
    x = []
    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            x.append(int(line.split()[index]))
    return x


if __name__ == '__main__':
    # 获取用户输入命令序列以及去重后的命令
    command_lists, fdist = load_user_command("./data/User3")
    # 对用户输入命令进行特征化
    command_feature = get_command_feature(command_lists, fdist)
    # 获取用户输入标签
    label = get_lable("./data/label", 2)
    # 恢复从 5000 开始的数据
    y = [0]*50 + label

    x_train = command_feature[0:100]
    y_train = y[0:100]

    x_test = command_feature[100:150]
    y_test = y[100:150]

    knn_clf = KNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(x_train, y_train)
    y_predict = knn_clf.predict(x_test)

    score = np.mean(y_test == y_predict)
    print(score)

