# 用户异常操作检测

使用 kNN 对用户的异常操作进行检测，数据集来自[于此](http://www.schonlau.net/)。
直接运行 attackDetection.py 即可，控制台输出的位 kNN 对每个用户预测的准确率。

程序运行时首先随机选取一个用户的操作进行训练，进行网格搜索获取最好的分类器参数，然后使用这个最好的分类器来训练所有的用户，得到预测值。

## 数据集

> Masquerade Data: This contains 50 files, one each for 50 users. Each file contains 15000 lines. Each line has one command.

用户操作数据包括 50 个文件，分别对应 50 个用户的操作。每个文件包含 15000 行，每一行包含有一个命令。

> Location of masquerades: This file contains 100 rows and 50 columns. Each column corresponds to one of the 50 users. Each row corresponds to a set of 100 commands, ** starting with command 5001 and ending with command 15000 ** . The entries in the files are 0 or 1. 0 means that the corresponding 100 commands are not contaminated by a masquerader. 1 means they are contaminated.

标签文件包含 100 行，50 列。每一列对应一个用户的所有操作，每一行对应 100 个命令的异常与否，对应的命令个数从 5001 开始到 15000 结束。整个文件由 0 和 1.0 构成，分别代表 100 个命令不包含异常操作和包含异常操作。