# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:04:08 2022

@author: sunxiaoyu
"""

# In[1]:
import numpy as np
from sklearn.svm import SVR  # SVM中的回归算法
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV  # 最佳参数网格搜索
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
from sklearn import ensemble
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.linear_model import Ridge


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
# In[2]:
data_train = pd.read_csv('new training set.csv', header=0, index_col=False, encoding="ANSI")
X_train = data_train.iloc[:, :-2]
state_train = data_train.iloc[:, -2]
margin_train = data_train.iloc[:, -1]
data_val = pd.read_csv('new validation set.csv', header=0, index_col=False, encoding="ANSI")
X_val = data_val.iloc[:, :-2]
state_val = data_val.iloc[:, -2]
margin_val = data_val.iloc[:, -1]
X_test_week1 = pd.read_csv('test_set_week1.csv', header=0, index_col=False, encoding="ANSI")
result_week1 = pd.read_csv('pred_answer_week1.csv', header=0, index_col=False, encoding="ANSI")

# In[3]
# class_clf=SGDClassifier(random_state=42,loss='hinge',penalty='elasticnet')
# class_clf=RandomForestClassifier()
class_clf = MLPClassifier(activation='relu', solver='adam', max_iter=10000, hidden_layer_sizes=(200, 100))
class_clf.fit(X_train, state_train)
state_pred_val = class_clf.predict(X_val)
recall = recall_score(state_val, state_pred_val)
precision = precision_score(state_val, state_pred_val)
F1 = 2 / (1 / recall + 1 / precision)
print('F1=%.4f' % F1)
print(confusion_matrix(state_val, state_pred_val))
# In[4]
# clf = DecisionTreeRegressor()
# clf=neighbors.KNeighborsRegressor()
# clf=ensemble.RandomForestRegressor(n_estimators=20)
clf = Ridge(alpha=2.0, fit_intercept=True)
clf = linear_model.LinearRegression()
clf = MLPRegressor()
clf.fit(X_train, margin_train)
margin_pred_val = clf.predict(X_val)
perf = 1 / (1 + np.sqrt(np.sum((margin_pred_val - margin_val.values) ** 2) / np.sum(margin_val.values ** 2)))
print('perf=%.4f' % perf)

# In[5]

steady_train = np.array([i for i, x in enumerate(state_train.values) if x == 0])
ste_val = np.array([i for i, x in enumerate(state_pred_val) if x == 1])
for i in [1e-7, 1e-8, 1e-9, 1e-10, 0]:
    clf = Ridge(alpha=i, fit_intercept=True)
    clf.fit(X_train.iloc[steady_train, :], margin_train.iloc[steady_train, :])
    margin_pred_val = clf.predict(X_val)
    margin_pred_val[ste_val] = 0
    u = np.array([i for i, x in enumerate(margin_pred_val) if x < 0])
    margin_pred_val[u] = 0
    Rg = 1 / (1 + np.sqrt(np.sum((margin_pred_val - margin_val.values) ** 2) / np.sum(margin_val.values ** 2)))
    print('Rg=%.4f' % Rg)
    perf = (F1 + Rg) / 2
    print('perf=%.4f' % perf)

# In[6]
state_test_week1 = class_clf.predict(X_test_week1)
margin_test_week1 = clf.predict(X_test_week1)
ste_test = np.array([i for i, x in enumerate(state_test_week1) if x == 1])
margin_test_week1[ste_test] = 0
t = np.array([i for i, x in enumerate(margin_test_week1) if x < 0])
margin_test_week1[t] = 0
state_test_week1[t] = 1
# In[7]
result = pd.DataFrame([i + 1 for i in range(9551)], columns=['id'])
result.insert(1, 'ret1', state_test_week1)
result.insert(2, 'ret2', margin_test_week1)
result.to_csv('result_week1.csv', index=False)
