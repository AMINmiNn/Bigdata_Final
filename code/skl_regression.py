from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# %%:数据导入
# 训练集
dataset1 = pd.read_csv("training set.csv", encoding='ANSI')
values1 = dataset1.values
train_X = values1[:, :-2]
# scaler1 = MinMaxScaler()
# train_X = scaler1.fit(train_X).transform(train_X)
train1_y = values1[:, -2]
train2_y = values1[:, -1]

# 验证集
dataset2 = pd.read_csv("validation set.csv", encoding='ANSI')
values2 = dataset2.values
validation_X = values2[:10000, :-2]
# scaler2 = MinMaxScaler()
# validation_X = scaler2.fit(validation_X).transform(validation_X)
validation1_y = values2[:10000, -2]
validation2_y = values2[:10000, -1]

# 测试集
test_X = values2[10000:, :-2]
# scaler3 = MinMaxScaler()
# test_X = scaler3.fit(test_X).transform(test_X)
test1_y = values2[10000:, -2]
test2_y = values2[10000:, -1]

# 竞赛测试集
dataset3 = pd.read_csv("test_set_week1.csv", encoding='ANSI')
values3 = dataset3.values
competition_X = values3[:, :]

# 回归训练
# %%线性模型
clf = linear_model.LinearRegression()
clf.fit(train_X, train2_y)
margin_pred = clf.predict(test_X)
# %%
# 评估回归精度
n = len(margin_pred)
sigma1 = 0
sigma2 = 0
for i in range(n):
    sigma1 += (margin_pred[i] - test2_y[i]) ** 2
    sigma2 += test2_y[i] ** 2
error = np.sqrt(sigma1 / sigma2)
Rg_margin = 1 / (1 + error)
print('Rg_margin=%.4f' % Rg_margin)
MSE = mean_squared_error(test2_y, margin_pred)
print('MSE=%.4f' % MSE)
