# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 09:13:34 2022

@author: 18338
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, f1_score  # 计算混淆矩阵
from sklearn.metrics import ConfusionMatrixDisplay
import sklearn.metrics  # 评估工具
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.neural_network import MLPClassifier  # MLP分类法
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from keras.models import load_model

# %%划分训练集和测试集
train_dataset = pd.read_csv('training set.csv', encoding='ANSI')
train_values = train_dataset.values
train_X = train_values[:, 0:452]
train_y = train_values[:, -2]
train_y2 = train_values[:, -1]
validation_dataset = pd.read_csv('validation set.csv', encoding='ANSI')
valid_values = validation_dataset.values
valid_X = valid_values[0:17472, :-2]
valid_y = valid_values[0:17472, -2]

X_test_week1 = pd.read_csv('test_set_week1.csv', header=0, index_col=False, encoding="ANSI")
X_test = X_test_week1.values

dataset3 = pd.read_csv("new test_set_week1.csv", encoding='ANSI')
values3 = dataset3.values
competition_X = values3[:, :]
"""scaler = StandardScaler()# 标准化
scaler.fit(train_X)
train_X = scaler.transform(train_X)
valid_X = scaler.transform(valid_X)"""

# %% 最佳参数搜索 随机森林
"""classfier = RandomForestClassifier(max_features = "sqrt")
n_estimators = [1,10,100,1000]
n_estimators = dict(n_estimators=n_estimators)
gsearch1 = GridSearchCV(classfier,n_estimators, cv=5, verbose=2, scoring="f1")
gsearch1.fit(train_X, train_y.ravel())

means = gsearch1.cv_results_['mean_test_score']
params = gsearch1.cv_results_['params']
for mean, param in zip(means, params):
   print("%f  with:   %r" % (mean, param))

print(gsearch1.best_params_)
print(gsearch1.best_score_)"""
# %%最佳参数搜索 MLP
"""classifier1 = MLPClassifier(activation='relu', solver='adam',alpha=0.1, max_iter=10000, hidden_layer_sizes=(190, 95))
ran = [1,10,100,1000,10000,100000,1000000]
param_grid = dict(random_state=ran)
gsearch2 = GridSearchCV(classifier1, param_grid, cv=3, verbose=2, scoring="f1")
gsearch2.fit(train_X, train_y.ravel())
means = gsearch2.cv_results_['mean_test_score']
params = gsearch2.cv_results_['params']
for mean, param in zip(means, params):
   print("%f  with:   %r" % (mean, param))

print(gsearch2.best_params_)
print(gsearch2.best_score_)"""
# %% 利用搜得的最佳参数进行分类 随机森林 搜得n_estimators=1000
"""model = RandomForestClassifier(n_estimators=2000,max_features="sqrt")
model.fit(train_X,train_y)
y_pred=model.predict(valid_X)
cm = confusion_matrix(valid_y, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()"""
# %% 利用搜得的最佳参数进行分类 MLP 搜得alpha=0.1
model1 = MLPClassifier(activation="relu", solver="adam", alpha=0.1, max_iter=10000, hidden_layer_sizes=(190, 95),
                       random_state=10)

model1.fit(train_X, train_y)
y_pred1 = model1.predict(valid_X)
# cm1 = confusion_matrix(valid_y, y_pred1)
# cm1_display = ConfusionMatrixDisplay(cm1).plot()
# plt.show()
# %% SVC分类 参数来自网络
"""model2 = SVC(kernel='rbf', C=250, gamma=1.0)
model2.fit(train_X,train_y)
y_pred2 = model2.predict(valid_X)
cm2 = confusion_matrix(valid_y, y_pred1)
cm2_display = ConfusionMatrixDisplay(cm2).plot()
plt.show()"""
# %%计算f1
# f1_micro = f1_score(valid_y, y_pred, average='micro')
# f1_micro1 = f1_score(valid_y, y_pred1, average='micro')
# f1_micro2 = f1_score(valid_y,y_pred2,average='micro')
recall = recall_score(valid_y, y_pred1)
precision = precision_score(valid_y, y_pred1)
F1 = 2 / (1 / recall + 1 / precision)
print("F1 = %.4f" % F1)
# %% 线性回归
"""reg4 = MLPRegressor(activation="relu",solver="adam",alpha=0.1,max_iter=10000,hidden_layer_sizes=(190,95),random_state=10).fit(train_X,train_y2)
y_pred4 = reg4.predict(valid_X)
perf4=1/(1+np.sqrt(np.sum((y_pred4-valid_y)**2)/np.sum(valid_y**2)))
print('perf4 for Linear=%.4f'%perf4)"""
# %%
state_test_week1 = model1.predict(X_test)
competition_X = np.column_stack((competition_X, state_test_week1))
model = load_model('regression_model.h5')
margin_test_week1 = model.predict(competition_X)
# margin_test_week1=reg4.predict(X_test)
ste_test = np.array([i for i, x in enumerate(state_test_week1) if x == 1])
margin_test_week1[ste_test] = 0
t = np.array([i for i, x in enumerate(margin_test_week1) if x <= 0])
margin_test_week1[t] = 0
state_test_week1[t] = 1
# In[7]
result = pd.DataFrame([i + 1 for i in range(9551)], columns=['id'])
result.insert(1, 'ret1', state_test_week1)
result.insert(2, 'ret2', margin_test_week1)
result.to_csv('result_week1_5.csv', index=False)
