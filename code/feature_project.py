import numpy as np
import pandas as pd

# %%:数据导入
# 训练集
dataset1 = pd.read_csv("training set.csv", encoding='ANSI')
values1 = dataset1.values
Q_percent = []
P_percent = []
for i in range(54):
    P_percent.append(values1[:, i + 344] - values1[:, i])
    Q_percent.append(values1[:, i + 398] - values1[:, i + 54])
for i in range(54):
    dataset1.insert(i+452, '{}P_percent'.format(i+1), P_percent[i])
for i in range(54):
    dataset1.insert(i+506, '{}Q_percent'.format(i+1), Q_percent[i])
dataset1.to_csv('new training set.csv', index=False)


# 验证集
dataset2 = pd.read_csv("validation set.csv", encoding='ANSI')
values2 = dataset2.values
Q_percent = []
P_percent = []
for i in range(54):
    P_percent.append(values2[:, i + 344] - values2[:, i])
    Q_percent.append(values2[:, i + 398] - values2[:, i + 54])
for i in range(54):
    dataset2.insert(i+452, '{}P_percent'.format(i+1), P_percent[i])
for i in range(54):
    dataset2.insert(i+506, '{}Q_percent'.format(i+1), Q_percent[i])
dataset2.to_csv('new validation set.csv', index=False)


# 测试集
dataset3 = pd.read_csv("test_set.csv", encoding='ANSI')
values3 = dataset3.values
Q_percent = []
P_percent = []
for i in range(54):
    P_percent.append(values3[:, i + 344] - values3[:, i])
    Q_percent.append(values3[:, i + 398] - values3[:, i + 54])
for i in range(54):
    dataset3.insert(i+452, '{}P_percent'.format(i+1), P_percent[i])
for i in range(54):
    dataset3.insert(i+506, '{}Q_percent'.format(i+1), Q_percent[i])
dataset3.to_csv('new test_set.csv', index=False)


# 竞赛测试集
dataset3 = pd.read_csv("test_set_week2.csv", encoding='ANSI')
values3 = dataset3.values
Q_percent = []
P_percent = []
for i in range(54):
    P_percent.append(values3[:, i + 344] - values3[:, i])
    Q_percent.append(values3[:, i + 398] - values3[:, i + 54])
for i in range(54):
    dataset3.insert(i+452, '{}P_percent'.format(i+1), P_percent[i])
for i in range(54):
    dataset3.insert(i+506, '{}Q_percent'.format(i+1), Q_percent[i])
dataset3.to_csv('new test_set_week2.csv', index=False)
