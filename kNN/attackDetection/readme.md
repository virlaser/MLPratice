# 用户异常操作检测

使用 kNN 对用户的异常操作进行检测

## 数据集

Masquerade Data: This contains 50 files, one each for 50 users. Each file contains 15000 lines. Each line has one command.

Location of masquerades: This file contains 100 rows and 50 columns. Each column corresponds to one of the 50 users. Each row corresponds to a set of 100 commands, ** starting with command 5001 and ending with command 15000 ** . The entries in the files are 0 or 1. 0 means that the corresponding 100 commands are not contaminated by a masquerader. 1 means they are contaminated.