# coding : utf-8

from nltk import FreqDist
from sklearn.model_selection import train_test_split, GridSearchCV
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
    # 2,8 切分数据集

    param_grid = [
        # 第一组搜索参数
        {
            'weights': ['uniform'],
            'n_neighbors': [i for i in range(1, 11)]
        },
        # 第二周搜索参数
        {
            'weights': ['distance'],
            'n_neighbors': [i for i in range(1, 11)],
            'p': [i for i in range(1, 6)]
        }
    ]
    # 数据划分
    x_train, x_test, y_train, y_test = train_test_split(command_feature, y, test_size=0.2)
    knn_clf = KNeighborsClassifier()
    # 网格搜索
    grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    print(grid_search.best_estimator_)
    # 最佳 kNN
    knn_clf = grid_search.best_estimator_
    score = knn_clf.score(x_test, y_test)
    print(score)

