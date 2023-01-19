import numpy as np
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score  # 计算混淆矩阵
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler  # 数据集规范化
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from keras import regularizers
from keras.models import load_model

# %%:数据导入

# 训练集
dataset1 = pd.read_csv("new training set.csv", encoding='ANSI')
values1 = dataset1.values
train_X = values1[:, :-2]
train1_y = values1[:, -2]
train2_y = values1[:, -1]
# 验证集
dataset2 = pd.read_csv("new validation set.csv", encoding='ANSI')
values2 = dataset2.values
validation_X = values2[:, :-2]
validation1_y = values2[:, -2]
validation2_y = values2[:, -1]
# 测试集
dataset3 = pd.read_csv("new test_set.csv", encoding='ANSI')
values3 = dataset3.values
test_X = values3[:, :-2]
test1_y = values3[:, -2]
test2_y = values3[:, -1]
# 竞赛测试集
dataset4 = pd.read_csv("new test_set_week2.csv", encoding='ANSI')
values4 = dataset4.values
competition_X = values4[:, :]
# %%
# 分类
# ====================================
model = tf.keras.Sequential()
input = train_X.shape[1]
# 隐藏层128
model.add(
    tf.keras.layers.Dense(256, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(input,)))
# tf.keras.layers.Dropout(0.1)
model.add(tf.keras.layers.Dense(256, kernel_regularizer=regularizers.l2(0), activation='relu'))
model.add(tf.keras.layers.Dense(256, kernel_regularizer=regularizers.l2(0.00), activation='relu'))
# tf.keras.layers.Dropout(0.1)
model.add(tf.keras.layers.Dense(256, kernel_regularizer=regularizers.l2(0), activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, ))
# model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(64, ))
# # model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, ))
# # model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Dense(64, ))
# model.add(tf.keras.layers.Dense(32))
# model.add(tf.keras.layers.Dense(16))
# Dropout层用于防止过拟合
# model.add(tf.keras.layers.Dropout(0.2))
# 隐藏层128
# model.add(tf.keras.layers.Dropout(0.2))
# 没有激活函数用于输出层，二分类问题，用sigmoid激活函数进行变换，多分类用softmax。
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation('sigmoid'))
# 使用高效的 ADAM 优化算法以，二分类损失函数binary_crossentropy，多分类的损失函数categorical_crossentropy
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()
# %%
callback_list = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.001,
        patience=500
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='classify_model2.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]

# %%
# 训练轮次
epochs = 100
# 批次数量
batch_size = 64

# ==============================
# 模型训练
# ==============================
history = model.fit(train_X,
                    train1_y,
                    shuffle=True,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[callback_list],
                    validation_data=(validation_X, validation1_y))
# %% 打印分类训练过程
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.show()
# %% 测试集预测
model = load_model('classify_model2.h5')
state_pred = model.predict(test_X)
for i in range(len(state_pred)):
    if state_pred[i] >= 0.43:
        state_pred[i] = 1
    else:
        state_pred[i] = 0
print(state_pred)
print(test1_y)

cm = confusion_matrix(test1_y, state_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()
accuracy = accuracy_score(test1_y, state_pred)
precision = precision_score(test1_y, state_pred)
recall = recall_score(test1_y, state_pred)
F1 = 2*precision*recall/(precision + recall)
print('F1=%.4f' % F1)
#%%
# %% ====================================
# 回归训练
# ====================================
model2 = tf.keras.Sequential()
input = train_X.shape[1]
# 隐藏层128
model2.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(input,)))
model2.add(tf.keras.layers.Dense(128, activation='relu'))
model2.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(256, activation='relu'))
model2.add(tf.keras.layers.Dense(1))
model2.compile(loss='mse', optimizer='adam', metrics=['mse'])
model2.summary()
# %%
# 训练轮次
epochs = 20
# 批次数量
batch_size = 128

# ==============================
# 模型训练
# ==============================
history2 = model2.fit(train_X,
                      train2_y,
                      shuffle=True,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(validation_X, validation2_y))


# %%打印回归训练过程


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'],
             label='Val Error')
    plt.legend()
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('mse')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train mse')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val mse')
    plt.legend()
    plt.show()


plot_history(history2)
# %% 测试集预测，以及结果初步处理

margin_pred = model2.predict(test_X)
margin_pred = margin_pred.flatten()
# 预测小于0的置为0
t = np.array([i for i, x in enumerate(margin_pred) if x < 0])
margin_pred[t] = 0
# 不稳定的置为0
ste_test = np.array([i for i, x in enumerate(state_pred) if x == 1])
margin_pred[ste_test] = 0
# %% 算训练评估指标
# Rg
n = len(margin_pred)
sigma1 = 0
sigma2 = 0
for i in range(n):
    sigma1 += (margin_pred[i] - test2_y[i]) ** 2
    sigma2 += test2_y[i] ** 2
error = np.sqrt(sigma1 / sigma2)
Rg_margin = 1 / (1 + error)
# MSE
MSE = mean_squared_error(test2_y, margin_pred)


# MAPE
def mean_absolute_percentage_error(real, predict):
    res = 0
    count = 0
    for i in range(len(real)):
        if real[i] != 0:
            res += abs((predict[i] - real[i]) / real[i])
            count += 1
    return res / count


MAPE = mean_absolute_percentage_error(test2_y, margin_pred)

print('Rg_margin=%.4f' % Rg_margin)
print('MAPE=%.4f' % MAPE)
print('MSE=%.4f' % MSE)

# %%
# ==========================
# ==========================
# 竞赛分类结果
competition1_y = model.predict(competition_X)
for i in range(len(competition1_y)):
    if competition1_y[i] >= 0.47:
        competition1_y[i] = 1
    else:
        competition1_y[i] = 0
print(competition1_y)
# %% 竞赛测试集回归结果
clf = linear_model.LinearRegression()
clf.fit(train_X, train2_y)
competition2_y = clf.predict(competition_X)
# 预测小于0的置为0
t = np.array([i for i, x in enumerate(competition2_y) if x < 0])
competition2_y[t] = 0
# 不稳定的置为0
ste_test = np.array([i for i, x in enumerate(competition1_y) if x == 1])
competition2_y[ste_test] = 0
# %%输出分类和回归结果
result = pd.DataFrame([i + 1 for i in range(len(competition1_y))], columns=['id'])
result.insert(1, 'ret1', competition1_y)
# result.insert(2, 'ret2', competition2_y)
result.to_csv('result_week2_1213.csv', index=False)
